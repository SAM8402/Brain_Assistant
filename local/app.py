from flask import Flask, request, jsonify, render_template_string, session
from flask_cors import CORS
import requests
import pickle
import sys
from datetime import datetime
from IPython.display import HTML
from openai import OpenAI
import ast
import asyncio
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from pathlib import Path
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.brain_region_assistant_langchain import UltraFastBrainAssistant

app = Flask(__name__)
app.secret_key = 'combined_brain_viewer_secret_key_change_in_production'  # Change this in production

# Enable CORS for all domains and all routes
CORS(app)

# Database Configuration
MySQL_db_user = "root"
MySQL_db_password = "Health#123"
MySQL_db_host = "apollo2.humanbrain.in"
MySQL_db_port = "3306"
MySQL_db_name = "HBA_V2"
MySQL_db = SQLDatabase.from_uri(f"mysql+pymysql://{MySQL_db_user}:{MySQL_db_password}@{MySQL_db_host}:{MySQL_db_port}/{MySQL_db_name}")

MySQL_DATABASE_URL = f"mysql+pymysql://{MySQL_db_user}:{MySQL_db_password}@{MySQL_db_host}:{MySQL_db_port}/{MySQL_db_name}"
MySQL_engine = create_engine(
    MySQL_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=21600
)

# Global caches
geojson_cache = {}
light_weight_viewer_cache = {}
thumbnail_cache = {}

# Base URLs
base_url = "http://dgx3.humanbrain.in:10603"

# Initialize the brain assistant
assistant = UltraFastBrainAssistant(use_web_search=True)

# ==================== Common Functions ====================

def MySQL_db_retriever(sql_query):
    with MySQL_engine.connect() as connection:  
        result = connection.execute(text(sql_query))  
    data = result.fetchall()
    return data

# ==================== Chatbot Functions ====================

def sync_assistant_history():
    """Sync Flask session history with the assistant's internal history"""
    chat_history = get_chat_history()
    
    # Clear and rebuild assistant's conversation history from session
    assistant.clear_conversation_history()
    
    # Add conversations to assistant's memory
    for msg in chat_history:
        if msg['type'] == 'user_query':
            # Find the corresponding assistant response
            next_msg = None
            idx = chat_history.index(msg)
            if idx + 1 < len(chat_history):
                next_msg = chat_history[idx + 1]
            
            if next_msg and next_msg['type'] in ['assistant_response', 'region_info']:
                assistant.add_to_conversation(
                    msg['content'],
                    next_msg['content'],
                    msg['type']
                )

def get_chat_history():
    """Get chat history from session"""
    if 'chat_history' not in session:
        session['chat_history'] = []
    return session['chat_history']

def add_to_chat_history(message_type, content, region=None):
    """Add a message to chat history"""
    chat_history = get_chat_history()
    message = {
        'type': message_type,  # 'user_query', 'assistant_response', 'region_info'
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'region': region
    }
    chat_history.append(message)
    session['chat_history'] = chat_history
    session.modified = True

def clear_chat_history():
    """Clear chat history from session"""
    session['chat_history'] = []
    session.pop('current_mode', None)  # Also clear stored mode
    session.modified = True
    # Also clear assistant's conversation history
    assistant.clear_conversation_history()

# ==================== Light Weight Viewer Functions ====================

def get_geojson(biosample, section):
    if biosample in geojson_cache.keys() and section in geojson_cache[biosample].keys():
        return geojson_cache[biosample][section]
    else:
        url = "http://gliacoder.humanbrain.in:8000/atlas/getAtlasgeoJson"
        headers = {"Content-Type": "application/json"}
        params = {
            "biosample": biosample,
            "sectionNo": section
        }
        print("biosample: ", biosample, "sectionNo: ", section)
        response = requests.get(url, headers=headers, json=params)
        if response.status_code != 200:
            return "Failed"
        elif response.status_code == 200 and response.json()['status'] == False :
            return "Biosample or Section Not Found!"

        data = response.json()['msg'][0]

        if biosample not in geojson_cache.keys():
            geojson_cache[biosample] = {}
        geojson_cache[biosample][section] = data

        return data

def get_metadata(biosample, section):
    query = f"""
    SELECT sct.rigidrotation , sct.width , sct.height, sct.jp2Path, ss.ontology, sct.trsdata, sct.export_status 
    FROM HBA_V2.seriesset ss
    JOIN HBA_V2.series s ON ss.id = s.seriesset
    JOIN HBA_V2.section sct ON s.id = sct.series
    WHERE ss.biosample = {biosample}
      AND s.seriestype = 1
      AND sct.positionindex = {section};
    """
    result = MySQL_db_retriever(query)
    jp2_file_name = result[0][3].split("/")[-1]
    
    if result[0][6] == 4:
        jp2_file_name = jp2_file_name.replace(".jp2", ".tif")

    result_dict = {"rotation":result[0][0], "width":result[0][1], "height":result[0][2],
     "jp2_file_name":jp2_file_name, "ontology":result[0][4], "trs_data":result[0][5]}

    return result_dict

def get_ontology_id(biosample):
    pass

def llm_resp(user_query): 
    client = OpenAI(base_url="http://dgx5.humanbrain.in:8999/v1", api_key="dummy")
    json_structure = {"biosample":"","section":""}

    prompt = """ 
    You are an intelligent system which will understand user query and take out the biosample and section from the query.

    Example:
    User: Show light weight viewer for biosample 222 and section 1000
    system: {'biosample':222,'section':1000}
    instruction: don't add anything else in the response follow the json reponse structure strictly given above in the example. 
    """ + f""" Json Structure {json_structure}"""

    completion = client.chat.completions.create(
        model="Llama-3.3-70B-Instruct",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_query}],
        temperature=0,
    )

    return completion.choices[0].message.content

def get_json_filename(biosample, section, ontology):
    path = F"/apps/analytics/{biosample}/appData/atlasEditor/{ontology}/NISL/{section}/"
    file_name = None
    for p in Path(path).rglob("*.json"):
        if "FlatTree" in p.parts[-1]:
            print(p.parts)
            file_name = p.parts[-1]
    return file_name
    
def imageInfo(biosample,section,jp2_file_name):
    imgUrl = f"https://apollo2.humanbrain.in/iipsrv/fcgi-bin/iipsrv.fcgi?FIF=/ddn/storageIIT/humanbrain/analytics/{biosample}/NISL/"+ jp2_file_name +"&WID=1024&GAM=1.4&MINMAX=1:0,255&MINMAX=2:0,255&MINMAX=3:0,255&JTL={{z}},{{tileIndex}}"
    print(imgUrl)
    return imgUrl

def geojsonUrl(biosample,section,ontology):
    json_file_name = get_json_filename(biosample, section, ontology)
    if json_file_name is None:
            json_file_name = ""
    jsonUrl  = f"https://apollo2.humanbrain.in/iipsrv/ddn/storageIIT/humanbrain/analytics/{biosample}/appData/atlasEditor/{ontology}/NISL/{section}/"+ json_file_name
    return jsonUrl

def get_light_weight_viewer_temp(biosample, section, stain_id=None, session_id=None):
    data = get_metadata(biosample, section)
    print("DATA:",data)
    height = data['height']
    width = data['width']
    rotation = data['rotation']
    jp2_file_name = data['jp2_file_name']
    ontology = data["ontology"]
    if data['trs_data']:
        trs_rot = ast.literal_eval(data['trs_data'])
        trs_rot = trs_rot['rotation'] 
    else:
        trs_rot = 0

    geourl = geojsonUrl(biosample,section, ontology)
    imgurl = imageInfo(biosample, section, jp2_file_name)
    print("INSIDE LIGHT WEIGHT VIEWER")
    print("IMAGE URL:", imgurl)
    print("GEO URL:", geourl)
    print("SESSION ID:", session_id)

    if biosample in light_weight_viewer_cache.keys():
        if section in light_weight_viewer_cache[biosample].keys():
            print("Rendering light weight viewer from cache")
            template = light_weight_viewer_cache[biosample][section]
    else:   
        template = """
        <!doctype html>
<html lang="en" style="height:100%;">

<head>
    <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/gh/openlayers/openlayers.github.io@master/en/v5.3.0/css/ol.css" type="text/css">
    <style>
        html,
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
        }

        .header {
            margin: 0;
            padding-left: 15px;
            color: black;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .header p {
            margin: 0;
            padding-top: 5px;
            font-size: 12px;
            font-weight: bold;
        }

        .name-container {
            display: flex;
            width: 100%;
            height: 0;
        }

        .feature-name {
            font-size: 20px;
            padding: 10px 40px;
            z-index: 3;
        }

        .map {
            height: 100%;
            width: 100%;
        }

        /* Control panel container */
        .control-panel {
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            flex-direction: column;
            gap: 8px;
            z-index: 1000;
        }

        /* Opacity slider styles */
        .slider-container {
            height: 120px;
            width: 18px;
            background-color: rgba(255, 255, 255, 0.6);
            border-radius: 4px;
            padding: 8px 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(0, 0, 0, 0.1);
        }

        #opacity-slider {
            writing-mode: bt-lr;
            -webkit-appearance: slider-vertical;
            appearance: slider-vertical;
            width: 3px;
            height: 90px;
            padding: 0;
            margin: 0;
        }

        /* Make slider thumb (handle) smaller */
        #opacity-slider::-webkit-slider-thumb {
            transform: scale(0.6);
            cursor: pointer;
        }

        /* For Firefox */
        #opacity-slider::-moz-range-thumb {
            transform: scale(0.6);
            cursor: pointer;
        }

        /* For IE/Edge */
        #opacity-slider::-ms-thumb {
            transform: scale(0.6);
            cursor: pointer;
        }

        .opacity-value {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 9px;
            margin-top: 3px;
        }

        /* Eye toggle button styles */
        .eye-toggle-container {
            width: 18px;
            height: 18px;
            background-color: rgba(255, 255, 255, 0.6);
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            cursor: pointer;
            transition: background-color 0.2s;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }

        .eye-toggle-container:hover {
            background-color: rgba(255, 255, 255, 0.8);
        }

        .eye-icon {
            width: 12px;
            height: 12px;
            transition: opacity 0.2s;
        }

        .eye-icon.hidden {
            opacity: 0.5;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/gh/openlayers/openlayers.github.io@master/en/v5.3.0/build/ol.js"></script>
    <title>Atlas Viewer</title>
</head>

<body style="height:100%;">
    <div class="header">"""+f"""
        <p>Atlas Viewer - Biosample: {biosample} Section:{section}</p>
        """+"""
    </div>
    <div class="name-container">
        <span class="feature-name" id="feature-name"></span>
    </div>
    <div id="map" class="map"></div>

    <!-- Control panel -->
    <div class="control-panel">
        <!-- Opacity slider control -->
        <div class="slider-container">
            <input type="range" min="0" max="99" value="50" class="slider" id="opacity-slider" orient="vertical">
            <div class="opacity-value" id="opacity-value">50</div>
        </div>

        <!-- Eye toggle button -->
        <div class="eye-toggle-container" id="eye-toggle" title="Toggle annotations visibility">
            <svg class="eye-icon" id="eye-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                <circle cx="12" cy="12" r="3"></circle>
            </svg>
        </div>
    </div>

    <script type="text/javascript">

        const Feature = ol.Feature;
        const Map = ol.Map;
        const View = ol.View;
        const GeoJSON = ol.format.GeoJSON;
        const Circle = ol.geom.Circle;
        const Point = ol.geom.Point;
        const Icon = ol.style.Icon;
        const TileLayer = ol.layer.Tile;
        const VectorLayer = ol.layer.Vector;
        const OSM = ol.source.OSM;
        const VectorSource = ol.source.Vector;
        const CircleStyle = ol.style.Circle;
        const Fill = ol.style.Fill;
        const Stroke = ol.style.Stroke;
        const Style = ol.style.Style;
        const Zoomify = ol.source.Zoomify;
        const Select = ol.interaction.Select;
        """ + f"""
        const geojsonUrl = '{geourl}'
        const mapurl = '{imgurl}'
        const mapsiz = [{width},{height}];
        const rigidrotation = {rotation};
        const biosample = '{biosample}';
        const section = '{section}';
        const tool_name= 'mini_atlas';
        const session_id = '{session_id}';
        const parsed_session_id = session_id;
        const trs_rot = {trs_rot};
        """ + """
        // Global variables
        let opacityAtlas = '50';
        let vectorSource;
        let selectInteraction;
        let annotationsVisible = true;
        let currentSelectedId = null;
        let vectorLayer;  // Global for access in toggle
        let map;  // Made global for render() access
        let notAvailableTimeout = null;  // Track timeout for "not available" message


        async function getGeoJSON(url) {
            try {
                const response = await fetch(url);
                return await response.json();
            } catch (error) {
                console.error('Error fetching GeoJSON:', error);
                return null;
            }
        }

        function styleFunction(feature) {
            if (typeof (opacityAtlas) == "number") {
                opacityAtlas = '50'
            }
            var data = feature.get('data')
            var clr = '#FF000070'

            if (data !== undefined && data.color_hex_triplet != undefined) {
                clr = data.color_hex_triplet;
                if (!data.color_hex_triplet.startsWith('#')) {
                    clr = '#' + clr
                }
            }

            if (clr.length == 7) {
                var atlasClr = clr;
                clr = clr + opacityAtlas;
            }

            var st = [
                new Style({
                    zIndex: -1,
                    stroke: new Stroke({
                        color: atlasClr,
                        width: 2.5,
                    }),
                    fill: new Fill({
                        color: clr
                    }),
                }),
            ];

            var geometry = feature.getGeometry();
            if (geometry.getType() === 'LineString') {
                var coordinates = geometry.getCoordinates();
                if (coordinates.length >= 2) {
                    var start = coordinates[coordinates.length - 2];
                    var end = coordinates[coordinates.length - 1];

                    var dx = end[0] - start[0];
                    var dy = end[1] - start[1];
                    var rotation = Math.atan2(dy, dx);

                    st.push(new Style({
                        stroke: new Stroke({
                            color: '#000000',
                            width: 3,
                        })
                    }));

                    st.push(new Style({
                        geometry: new Point(end),
                        image: new Icon({
                            src: 'https://apollo2.humanbrain.in/viewer/assets/images/colorsvg/right_arrow.svg',
                            anchor: [0.75, 0.5],
                            rotateWithView: true,
                            rotation: -rotation,
                        })
                    }));
                }
            }

            st.push(new Style({
                zIndex: -1,
                image: new CircleStyle({
                    radius: 5.9,
                    stroke: new Stroke({
                        color: atlasClr,
                        width: 1,
                    }),
                    fill: new Fill({
                        color: clr,
                    }),
                }),
                geometry: function (feature) {
                    var coordinates = feature.getGeometry().getCoordinates();
                    return new Point(coordinates);
                },
            }));

            return st;
        }

        // Update all features with the new opacity
        function updateLayerOpacity() {
            if (vectorSource) {
                vectorSource.getFeatures().forEach(feature => {
                    // Check if this feature is currently selected/highlighted
                    const data = feature.get('data');
                    const isHighlighted = currentSelectedId !== null && data && data.id === currentSelectedId;
                    
                    if (isHighlighted) {
                        // Keep the highlight style for selected features
                        feature.setStyle(new Style({
                            stroke: new Stroke({
                                color: 'red',
                                width: 3
                            }),
                            fill: new Fill({
                                color: 'transparent'
                            })
                        }));
                    } else {
                        // Apply normal style with updated opacity
                        feature.setStyle(styleFunction(feature));
                    }
                });
            }
        }

        // Function to trigger click event on feature by ID
       // Function to trigger click event on feature by ID
        function triggerClickEventById(targetId) {
            if (!vectorSource || !selectInteraction) {
                console.error('Vector source or select interaction not available');
                return;
            }

            // Clear existing selection first
            selectInteraction.getFeatures().clear();

            const featureNameEl = document.getElementById("feature-name");

            // If targetId is -1, reset all features to normal styling and clear feature name
            if (targetId === -1) {
                // Reset all features to their normal styling
                vectorSource.getFeatures().forEach(feature => {
                    feature.setStyle(styleFunction(feature));
                });

                // Clear the feature name display
                if (featureNameEl) {
                    featureNameEl.textContent = "";
                }

                console.log('Cleared all feature selections and reset to normal styling');
                return;
            }

            // Find features with matching ID
            const matchingFeatures = vectorSource.getFeatures().filter(f => {
                const data = f.get('data');
                return data && data.id === targetId;
            });

            // First, reset all features to normal styling (regardless of whether feature is found)
            vectorSource.getFeatures().forEach(feature => {
                feature.setStyle(styleFunction(feature));
            });

            if (matchingFeatures.length > 0) {
                // Add the first matching feature to selection
                selectInteraction.getFeatures().push(matchingFeatures[0]);

                // Highlight the matching features
                matchingFeatures.forEach(f => {
                    f.setStyle(new Style({
                        stroke: new Stroke({
                            color: 'red',
                            width: 3
                        }),
                        fill: new Fill({
                            color: 'transparent'
                        })
                    }));
                });
                
                // Set the name of the first matched region
                const featureData = matchingFeatures[0].get('data');
                if (featureData && featureData.name && featureNameEl) {
                    // Clear any pending "not available" timeout
                    if (notAvailableTimeout) {
                        clearTimeout(notAvailableTimeout);
                        notAvailableTimeout = null;
                    }
                    featureNameEl.textContent = ` ${featureData.name}`;
                    featureNameEl.style.color = ""; // Reset to default color
                }
                console.log(`Triggered click event for ${matchingFeatures.length} features with ID: ${targetId}`);
            } else {
                // Feature not found - deselect any current region and show "Not Available" message
                currentSelectedId = null;
                
                // Clear any existing selection and reset all features to normal styling
                if (selectInteraction) {
                    selectInteraction.getFeatures().clear();
                }
                
                // Reset all features to their normal styling
                vectorSource.getFeatures().forEach(feature => {
                    feature.setStyle(styleFunction(feature));
                });
                
                if (featureNameEl) {
                    // Clear any existing timeout first
                    if (notAvailableTimeout) {
                        clearTimeout(notAvailableTimeout);
                    }
                    
                    featureNameEl.textContent = "Region not available in this section";
                    featureNameEl.style.color = "#888"; // Make it gray to indicate it's not available
                    
                    // Set new timeout and store its reference
                    notAvailableTimeout = setTimeout(() => {
                        featureNameEl.textContent = "";
                        featureNameEl.style.color = ""; // Reset to default color
                        notAvailableTimeout = null; // Clear the reference
                    }, 2000);
                }
                console.log(`No features found with ID: ${targetId}`);
                }
            }
        
        // Call the function to get data
        getGeoJSON(geojsonUrl).then(geojsonData => {
            const geojsonObject = geojsonData;

            vectorLayer = new VectorLayer({  // Assigned to global
                transition: 0,
                source: new VectorSource({
                    format: new GeoJSON(),
                    wrapX: false,
                }),
                style: styleFunction,
            });

            const zoomifySource = new Zoomify({
                url: mapurl,
                size: mapsiz,
                crossOrigin: "anonymous",
                tierSizeCalculation: 'truncated',
                imageSmoothing: false,
                tileSize: 2048
            });

            const imagery = new TileLayer({
                source: zoomifySource
            });

            const extent = zoomifySource.getTileGrid().getExtent();

            map = new Map({  // Assigned to global
                layers: [imagery, vectorLayer],
                target: 'map',
                view: new View({
                    zoom: 10,
                    minZoom: 8,
                    maxZoom: 19,
                    rotation: (rigidrotation * Math.PI / 180),
                    extent: extent
                }),
                controls: ol.control.defaults({
                    rotate: false
                })
            });

            map.getView().fit(imagery.getSource().getTileGrid().getExtent());
            if (trs_rot != 0)
                map.getView().setRotation(trs_rot);
            var centerMap = map.getView().getCenter();
            vectorSource = vectorLayer.getSource();

            var features = vectorSource.getFormat().readFeatures(geojsonObject);
            features.forEach(element => {
                var elementRotate = element.getGeometry();
                var xy = centerMap;
                elementRotate = elementRotate.rotate(((rigidrotation) * Math.PI / 180), xy);
            });
            vectorSource.addFeatures(features);
            vectorSource.getFeatures().forEach(element => {
                element.setStyle(styleFunction(element));
            });


            selectInteraction = new Select({
                condition: ol.events.condition.singleClick,
                style: null, // Disable default OpenLayers selection style
            });

            map.addInteraction(selectInteraction);

            selectInteraction.on('select', async function (e) {
                // Reset all styles
                vectorSource.getFeatures().forEach(feature => {
                    feature.setStyle(styleFunction(feature));
                });

                const selected = e.selected;
                const featureNameEl = document.getElementById("feature-name");

                if (selected.length > 0) {
                    const clickedFeature = selected[0].getProperties();
                    const clickedId = clickedFeature.data.id;
                    const clickedName = clickedFeature.data.name;
                    
                    currentSelectedId = clickedId;

                    // Highlight all features with the same ID
                    const matchingFeatures = vectorSource.getFeatures().filter(f => {
                        const data = f.get('data');
                        return data && data.id === clickedId;
                    });

                    matchingFeatures.forEach(f => {
                        f.setStyle(new Style({
                            stroke: new Stroke({
                                color: 'red',
                                width: 3
                            }),
                            fill: new Fill({
                                color: 'transparent'
                            })
                        }));
                    });

                    // Clear any pending "not available" timeout when selecting a valid region
                    if (notAvailableTimeout) {
                        clearTimeout(notAvailableTimeout);
                        notAvailableTimeout = null;
                    }

                    featureNameEl.textContent = ` ${clickedName}`;
                    featureNameEl.style.color = ""; // Reset to default color

                    const payload = {
                        id: parsed_session_id,
                        tool_name: tool_name,
                        params: {
                            id: clickedId,
                            name: clickedName,
                            biosample: biosample,
                            section: section
                        }
                    };
                    window.parent.postMessage(payload, '*');

                    try {
                        const response = await fetch("https://llm.humanbrain.in:1062/context", {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json"
                            },
                            body: JSON.stringify(payload)
                        });
                        const data = await response.json();
                        console.log("Server Response:", data);
                    } catch (error) {
                        console.error("Error sending data to context API:", error);
                    }

                } else {
                    // Clicked outside any feature
                    currentSelectedId = null;
                    
                    // Clear any pending "not available" timeout when clicking outside
                    if (notAvailableTimeout) {
                        clearTimeout(notAvailableTimeout);
                        notAvailableTimeout = null;
                    }
                    
                    featureNameEl.textContent = "";
                    featureNameEl.style.color = ""; // Reset to default color

                    const payload = {
                        id: parsed_session_id,
                        tool_name: tool_name,
                        params: {
                            id: -1,
                            name: null,
                            biosample: biosample,
                            section: section
                        }
                    };
                    window.parent.postMessage(payload, '*');
                }
            });


            // Set up opacity slider interaction
            let lastOpacity = opacityAtlas;
            let throttleTimeout = null;

            const opacitySlider = document.getElementById("opacity-slider");
            const opacityValueDisplay = document.getElementById("opacity-value");

            function throttledOpacityUpdate(newOpacity) {
                if (newOpacity === lastOpacity) return;
                lastOpacity = newOpacity;

                opacityAtlas = newOpacity;
                opacityValueDisplay.textContent = opacityAtlas;
                updateLayerOpacity();

                //if (selectInteraction) {
               //     selectInteraction.getFeatures().clear();
               // }

                //document.getElementById("feature-name").textContent = "";
            }

            opacitySlider.addEventListener("input", function () {
                const newOpacity = this.value.toString().padStart(2, '0');

                if (throttleTimeout) clearTimeout(throttleTimeout);

                throttleTimeout = setTimeout(() => {
                    throttledOpacityUpdate(newOpacity);
                }, 10);
            });

            // Eye toggle functionality
            const eyeToggle = document.getElementById('eye-toggle');
            const eyeIcon = document.getElementById('eye-icon');

            eyeToggle.addEventListener('click', function() {
                annotationsVisible = !annotationsVisible;
                
                if (annotationsVisible) {
                    // Show annotations
                    vectorLayer.setVisible(true);
                    eyeIcon.innerHTML = `
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                        <circle cx="12" cy="12" r="3"></circle>
                    `;
                    eyeIcon.classList.remove('hidden');
                    
                    // Restore previous selection if any
                    if (currentSelectedId !== null) {
                        // Small delay to ensure layer is fully rendered
                        setTimeout(() => {
                            // Re-apply the highlight to all matching features
                            const matchingFeatures = vectorSource.getFeatures().filter(f => {
                                const data = f.get('data');
                                return data && data.id === currentSelectedId;
                            });
                            
                            if (matchingFeatures.length > 0) {
                                // Clear any existing selection
                                selectInteraction.getFeatures().clear();
                                
                                // Reset all features to normal styling first
                                vectorSource.getFeatures().forEach(feature => {
                                    feature.setStyle(styleFunction(feature));
                                    feature.changed();  // Force reset refresh
                                });
                                
                                // Add the first matching feature to selection
                                selectInteraction.getFeatures().push(matchingFeatures[0]);
                                
                                // Apply highlight style to all matching features
                                matchingFeatures.forEach(f => {
                                    f.setStyle(new Style({
                                        stroke: new Stroke({
                                            color: 'red',
                                            width: 3
                                        }),
                                        fill: new Fill({
                                            color: 'transparent'
                                        })
                                    }));
                                    f.changed();  // Force refresh each feature
                                });
                                
                                // Force source and layer refresh
                                vectorSource.changed();
                                vectorLayer.changed();
                                map.render();  // Force full map redraw
                                
                                // Update the feature name display
                                const featureData = matchingFeatures[0].get('data');
                                const featureNameEl = document.getElementById("feature-name");
                                if (featureData && featureData.name && featureNameEl) {
                                    featureNameEl.textContent = ` ${featureData.name}`;
                                }
                            }
                        }, 200);  // Increased timeout for rendering
                    }
                } else {
                    // Hide annotations
                    vectorLayer.setVisible(false);
                    eyeIcon.innerHTML = `
                        <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
                        <line x1="1" y1="1" x2="23" y2="23"></line>
                    `;
                    eyeIcon.classList.add('hidden');
                }
            });

        });

        // Updated message listener to trigger click events
        window.addEventListener('message', (event) => {
            console.log('Message received in iframe:', event.data);

            // Handle the existing 'sayHi' action
            if (event.data.action === 'sayHi') {
                alert('Got this from parent: ' + event.data.data);
            }

            // Handle highlight action with region data
            if (event.data.action === 'highlight' && event.data.region) {
                const { id } = event.data.region;
                console.log(`Triggering click event for region ID: ${id}`);
                triggerClickEventById(id);
            }

            // Handle direct region data (your format: {id: 158, name: 'Ganglionic eminence', acronym: 'GE'})
            if (event.data.id && event.data.name && event.data.type == "HighlightRegion" && typeof event.data.id === 'number') {
                console.log(`Triggering click event for region: ${event.data.name} (ID: ${event.data.id})`);
                triggerClickEventById(event.data.id);
            }
        });

    </script>
</body>

</html>
        """
    return template

def get_light_weight_viewer(user_query):
    resp = llm_resp(user_query)
    resp = ast.literal_eval(resp)
    biosample = resp['biosample']
    section  = resp['section']
    data = get_geojson(biosample, section)
    height = data['height']
    width = data['width']
    rotation = data['rotation']
    geojson = data['geoJson']
    geourl = geojsonUrl(biosample,section)
    imgurl = imageInfo(biosample,section)

    template = """<!doctype html>
                  <html lang="en" style="height:100%;">
                  <head>
                      <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/openlayers/openlayers.github.io@master/en/v5.3.0/css/ol.css" type="text/css">
                      <style>
                          .header {
                              display: flex;
                              width: 100%;
                              justify-content: space-between;
                          }
                          .feature-name{
                              font-size: 20px;
                              padding-top: 20px;
                          }
                        .map {
                          height:100%;
                          width:100%;
                        }
                      </style>
                      <script src="https://cdn.jsdelivr.net/gh/openlayers/openlayers.github.io@master/en/v5.3.0/build/ol.js"></script>
                      <title>Atlas Viewer</title>
                      </head>
                      <body style="height:100%;">
                              <div class="header">
                                  <h2>Biosample: 222 Section:199</h2>
                                  <span class="feature-name" id="feature-name"></span>
                              </div>
                              <div id="map" class="map"></div>
                              <script type="text/javascript">

                              const Feature = ol.Feature;
                              const Map = ol.Map;
                              const View = ol.View;
                              const GeoJSON = ol.format.GeoJSON;
                              const Circle = ol.geom.Circle;
                              const Point = ol.geom.Point;
                              const Icon = ol.style.Icon;
                              const TileLayer = ol.layer.Tile;
                              const VectorLayer = ol.layer.Vector;
                              const OSM = ol.source.OSM;
                              const VectorSource = ol.source.Vector;
                              const CircleStyle = ol.style.Circle;
                              const Fill = ol.style.Fill;
                              const Stroke = ol.style.Stroke;
                              const Style = ol.style.Style;
                              const Zoomify = ol.source.Zoomify;
                              const Select = ol.interaction.Select;

                              """ + f"""

                                const geojsonUrl = '{geourl}'
                                const mapurl = '{imgurl}'
                                const mapsiz = [{width},{height}];
                                const rigidrotation = {rotation};

                              """ + """

                              async function getGeoJSON(url) {
                                  try {
                                      const response = await fetch(url);
                                      return await response.json();
                                  } catch (error) {
                                      console.error('Error fetching GeoJSON:', error);
                                      return null;
                                  }
                              }

                              function styleFunction(feature, opacityAtlas = '99') {
                                  if(typeof(opacityAtlas) == "number") {    
                                  opacityAtlas = '99'
                                  }
                                  var data = feature.get('data')
                                  var clr = '#FF000070'

                                  if (data !== undefined && data.color_hex_triplet != undefined){
                                      clr =  data.color_hex_triplet;
                                      if(!data.color_hex_triplet.startsWith('#')){
                                      clr= '#'+clr
                                      }
                                  }

                                  if(clr.length==7) {
                                      var atlasClr = clr;
                                      clr = clr + opacityAtlas;
                                  }

                                  var st = [
                                      new Style({
                                      zIndex: -1,
                                      stroke: new Stroke({
                                          color: atlasClr,
                                          width: 2.5,
                                      }),
                                      fill: new Fill({
                                          color: clr
                                      }),
                                      }),
                                  ];

                                  var geometry = feature.getGeometry();
                                  if (geometry.getType() === 'LineString') {
                                      var coordinates = geometry.getCoordinates();
                                      if (coordinates.length >= 2) {
                                      var start = coordinates[coordinates.length - 2];
                                      var end = coordinates[coordinates.length - 1];
                                      
                                      var dx = end[0] - start[0];
                                      var dy = end[1] - start[1];
                                      var rotation = Math.atan2(dy, dx);

                                      st.push(new Style({
                                              stroke: new Stroke({
                                              color: '#000000',
                                              width: 3,
                                              })
                                          }));
                                      
                                      st.push(new Style({
                                          geometry: new Point(end),
                                          image: new Icon({
                                          src: 'https://apollo2.humanbrain.in/viewer/assets/images/colorsvg/right_arrow.svg',
                                          anchor: [0.75, 0.5],
                                          rotateWithView: true,
                                          rotation: -rotation,
                                          })
                                      }));
                                      } 
                                  }
                                  
                                  st.push(new Style({
                                      zIndex: -1,
                                      image: new CircleStyle({
                                      radius: 5.9,
                                      stroke: new Stroke({
                                          color: atlasClr,
                                          width: 1,
                                      }),
                                      fill: new Fill({
                                          color: clr,
                                      }),
                                      }),
                                      geometry: function (feature) {
                                      var coordinates = feature.getGeometry().getCoordinates();
                                      return new Point(coordinates);
                                      },
                                  }));

                                  return st;
                              }

                              // Call the function to get data
                              getGeoJSON(geojsonUrl).then(geojsonData => {
                                  const geojsonObject = geojsonData;

                                  const vectorLayer = new VectorLayer({
                                      transition: 0,
                                      source: new VectorSource({
                                          format: new GeoJSON(),
                                          wrapX: false,
                                      }),
                                      style: styleFunction,
                                  });

                                  const zoomifySource = new Zoomify({
                                      url: mapurl,
                                      size: mapsiz,
                                      crossOrigin: "anonymous",
                                      tierSizeCalculation: 'truncated',
                                      imageSmoothing: false,
                                      tileSize: 2048
                                  });

                                  const imagery = new TileLayer({
                                      source: zoomifySource
                                  });

                                  const map = new Map({
                                      layers: [imagery, vectorLayer],
                                      target: 'map',
                                      view: new View({
                                          zoom: 10,
                                          rotation: (rigidrotation * Math.PI / 180)
                                      }),
                                  });

                                  map.getView().fit(imagery.getSource().getTileGrid().getExtent());
                                  var centerMap = map.getView().getCenter();
                                  vectorSource = vectorLayer.getSource();

                                  var features = vectorSource.getFormat().readFeatures(geojsonObject);
                                  features.forEach(element => {
                                      var elementRotate = element.getGeometry();
                                      var xy = centerMap;
                                      elementRotate = elementRotate.rotate((( rigidrotation) * Math.PI / 180), xy);
                                  });
                                  vectorSource.addFeatures(features);
                                  vectorSource.getFeatures().forEach(element => {
                                      element.setStyle(styleFunction);
                                  });

                                  
                                  const selectInteraction = new Select({
                                      condition: ol.events.condition.singleClick,
                                      style: null, // Disable default OpenLayers selection style
                                  });

                                  map.addInteraction(selectInteraction);

                                  selectInteraction.on('select', function (e) {
                                      vectorSource.getFeatures().forEach(feature => {
                                          feature.setStyle(styleFunction(feature));  
                                      });

                                      e.selected.forEach(feature => {
                                          feature.setStyle(new Style({
                                              stroke: new Stroke({
                                                  color: 'red',  
                                                  width: 3
                                              }),
                                              fill: new Fill({
                                                  color: 'transparent'
                                              })
                                          }));
                                      });

                                      if (e.selected.length > 0) {
                                          let feature = e.selected[0].getProperties();                                          
                                          document.getElementById("feature-name").textContent = ` ${feature.data.name}`;
                                      } else {
                                          document.getElementById("feature-name").textContent = "";
                                      }
                                  });
                              });


                              window.addEventListener('message', (event) => {
                            // You can check event.origin here for security
                            console.log('Message received in iframe:', event.data);

                            if (event.data.action === 'sayHi') {
                                alert('Got this from parent: ' + event.data.data); 
                            }
                            });
                          </script>
                      </body>
                  </html>
              """

    with open("/apps/src/utils/agent/tools/actions/light_weight_viewer/light_weight_viewer.html", "w") as h:
      h.write(template)

    return "http://dgx3.humanbrain.in:10605/ol"

# ==================== Thumbnail Viewer Functions ====================

def get_ssid(biosample_id):
    query = f"SELECT * FROM HBA_V2.seriesset where biosample={biosample_id};"
    result = MySQL_db_retriever(query)
    return result[0][0]

def get_stain_map(mnemonic_name):
    result = MySQL_db_retriever(f"""
    SELECT name FROM HBA_V2.seriestype where mnemonic='{mnemonic_name}';
    """)
    return result[0][0]

def llm_resp_thumbnail(user_query, chat_history=None, page_context=None):
    client = OpenAI(base_url="http://dgx5.humanbrain.in:8999/v1", api_key="dummy")
    json_structure = {"biosample":"","series":""}
    print("PAGE CONTEXT INSDE GET THUMBNAILS", page_context)
    prompt_template = """ 
    <|begin_of_text|><|start_header_id|>SYSTEM<|end_header_id|>
    You are an intelligent system which will understand user query and take out the biosample and section from the query.

    Instruction: 
    - don't add anything else in the response follow the json reponse structure strictly given above in the example. 
    - these are the histological stains i.e series types available:
    - if the user query doesn't mention any biosample or series search for the required params in Chat history and page context provided
    - here are the stains present by their names an mnemonics: NISSL - NISL, Haematoxylin and Eosin - HEOS, Block face image - BFI, Cell Detections - CDS, Immuno Histo Chemistry - IHCS, BFI White - BFIW, MRI, ATLAS - ATLA, MYELIN - MYEL, IHC7[CalB] - IHC7, IHC1[NeuN] - IHC1, IHC2[TH] - IHC2, IHC5[CalR] - IHC5, IHC6[PARV] - IHC6, IHC8[CHAT] - IHC8, IHC4[VGluT2-NFH] - IHC4, MT, IHC11[bAmy] - IH11, IHC_type_17 - IH17
    - use the mnemonic name has the series type in the params   
    - if the user doesn't mention any stain or series it should defaulty map to 'NISL'

    [EXAMPLE BEGINS HERE]
    Example:
    User: open thumbnails of brain 222
    page context:{'ssid': 100, 'seriesType': 'NISL', 'secid': 916, 'biosampleId': '222'}
    system: {'biosample':222,'series':'NISL', 'roi_section':916}

    User: open thumbnails of brain 244
    page context:{'ssid': 100, 'seriesType': 'NISL', 'secid': 916, 'biosampleId': '244'}
    system: {'biosample':244,'series':'NISL', 'roi_section':916}
    note: in the above page context the biosample is different, then also it should take the roi section from that
    
    User: fb85 thumbnails
    system: {'biosample':244,'series':'NISL', 'roi_section':916}
    note: in the above page context the biosample is different, then also it should take the roi section from that
    
    User: open thumbnails of brain 222
    page_context:{} or None
    system: {'biosample':222,'series':'NISL', 'roi_section':1}

    User:open thumbnails of H&E brain 222 
    page_context:{} or None
    system: {'biosample':222,'series':'HEOS', 'roi_section':1}
    [EXAMPLE ENDS HERE]
    
    """+f"""
    CHAT HISTORY:
    {chat_history}
    
    OUTPUT JSON STRUCTURE: {json_structure}
    
    PAGE CONTEXTt: {page_context}
    
    |eot_id|><|start_header_id|>USER<|end_header_id|>
    {user_query}<|eot_id|>"""
    
    messages = [
        {"role": "system", "content": prompt_template}]

    completion = client.chat.completions.create(
        model="Llama-3.3-70B-Instruct",
        messages=messages,
        temperature=0,
    )

    return completion.choices[0].message.content

def thumbnail_viewer(query, session_id, chat_history=None, page_context=None):
    print("thumbnail viewer")
    
    print("query:", query)
    print(page_context)
    llm_reponse = llm_resp_thumbnail(query, chat_history, page_context)
    print("THUMBNAIL VIEWER LLM RESPONSE:", llm_reponse)
    response = ast.literal_eval(llm_reponse)

    print("THUMBNAIL VIEWER RESPONSE:", response)
    biosample, series_type, roi_section = response['biosample'], response['series'], response['roi_section']

    return f"{base_url}/thumbnail/viewer/{biosample}/{series_type}/{session_id}/{roi_section}"

def get_thumbnail_viewer(biosample, series_type, session_id, roi_section=1):
    print("biosample: ", biosample, "series_type: ", series_type, "session_id: ", session_id, "roi_section: ", roi_section)
    ss_id = get_ssid(biosample)
    stain_name = get_stain_map(series_type)
    print(ss_id)
    template = """<!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Thumbnail Viewer</title>
            <style>
                html, body {
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
            }

            .header{
                margin: 0;
                padding-left: 15px;
                color: black;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .header p{
                margin: 0;
                padding-top: 5px;
                font-size: 12px;
                font-weight: bold;
            }

            .image-grid-container {
                width: 100%;
                height: 100%;
                border: none;
            }

            .image-grid {
                display: grid;
                gap: 5px;
                width: 100%;
                height: 100%;
                padding: 10px;
            }

            .container {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 20px;
                margin-bottom: 10px;
            }

            .image-grid-wrapper {
                display: flex;
                width: 200%;
                transition: transform 0.5s ease-in-out;
            }

            .image-container {
                position: relative;
                border: 2px solid #444444;
                padding: 5px;
                cursor: pointer;
                transition: transform 0.2s;
                background-color: #f8f8f8;
                display: flex;
                justify-content: center;
                align-items: center;
                width: 100px;
                height: 100px;
            }

            .green-dot {
                width: 10px;
                height: 10px;
                background-color: green;
                border-radius: 50%;
                position: absolute;
                top: 5px;
                right: 5px;
            }

            .image-container:hover {
                transform: scale(1.1);
            }

            .image-container img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }

            .image-id {
                position: absolute;
                bottom: 5px;
                left: 5px;
                color: red;
                font-size: 12px;
                font-weight: bold;
            }

            .nav-buttons {
                padding: 5px 10px;
                font-size: 16px;
                cursor: pointer;
                border: none;
                background-color: #eeeeee;
                color: black;
                border: 1px solid black;
                border-radius: 3rem;
                position: absolute;
                top: 50%;
                transform: translateY(-50%);
                z-index: 2;
            }

            #prev {
                left: 10px;
            }

            #next {
                right: 10px;
            }

            .nav-buttons:disabled {
                opacity: 0.3;
                cursor: not-allowed;
            }
            .selected {
                border: 2px solid red !important;
            }
            </style>
        </head>
        <body>
            <div class="header">"""+f"""
                <p>Thumbnail Viewer - Biosample: {biosample} </p>"""+"""
            </div>

            <div class="container">
                <button id="prev" class="nav-buttons" disabled>&#11164;</button>
                <div class="image-grid-container">
                    <div class="image-grid-wrapper">
                        <div id="image-grid-1" class="image-grid"></div>
                        <div id="image-grid-2" class="image-grid"></div>
                    </div>
                </div>
                <button id="next" class="nav-buttons">&#11166;</button>
            </div>

            <script>
                """+f"""
                const API_URL = "https://llm.humanbrain.in:1062/get/brain/thumbnails/{ss_id}";
                const session_id = "{session_id}";
                const biosample = "{biosample}";
                roi_section = {roi_section};
                const targetSeriesName = "{stain_name}";   //---------------series type--------

                """+"""
                let imagesData = [];
                let startIndex = 0;
                let isAnimating = false;

                const gridWrapper = document.querySelector(".image-grid-wrapper");
                const grid1 = document.getElementById("image-grid-1");
                const grid2 = document.getElementById("image-grid-2");
                const prevButton = document.getElementById("prev");
                const nextButton = document.getElementById("next");

                async function fetchImages() {
                    try {
                        const response = await fetch(API_URL);
                        if (!response.ok) {
                            throw new Error(`HTTP error! Status: ${response.status}`);
                        }
                        const data = await response.json();
                        const selectedSeries = data.find(item => item.seriesType === targetSeriesName);
                        imagesData = selectedSeries?.thumbnails ?? [];
                        if (imagesData.length === 0) {
                            console.warn("No images found for the selected series.");
                            return;
                        }
                        
                        // Find index of roi_section or closest section
                        let roiIndex = imagesData.findIndex(img => img.sectionNo == roi_section);

                        if (roiIndex === -1) {
                            // Not found exact, find closest section
                            let closestDiff = Infinity;
                            let closestIndex = 0;
                            for (let i = 0; i < imagesData.length; i++) {
                                const diff = Math.abs(imagesData[i].sectionNo - roi_section);
                                if (diff < closestDiff) {
                                    closestDiff = diff;
                                    closestIndex = i;
                                }
                            }
                            roiIndex = closestIndex;
                        }

                        // Calculate startIndex so roiIndex is roughly in the middle of the grid
                        const columns = calculateColumns();
                        const rows = Math.ceil(window.innerHeight / 150);
                        const imagesPerPage = columns * rows;

                        startIndex = roiIndex - Math.floor(imagesPerPage / 2);
                        if (startIndex < 0) startIndex = 0;
                        if (startIndex > imagesData.length - imagesPerPage) {
                            startIndex = Math.max(imagesData.length - imagesPerPage, 0);
                        }

                        renderImages(grid1, startIndex);
                        updatePaginationButtons();
                    } catch (error) {
                        console.error("Error fetching images:", error);
                    }
                }


                function calculateColumns() {
                    const containerWidth = document.querySelector(".image-grid-container").clientWidth;
                    return Math.floor(containerWidth / 110); // Assuming column width ~110px
                }

                function calculateRows() {
                    return Math.floor(window.innerHeight / 110); // Assuming row height ~150px
                }

                function preloadImages() {
                    imagesData.forEach(image => {
                        const img = new Image();
                        img.src = image.thumbnailUrl;
                    });
                }

                function createImageElement(image, index) {
                    const container = document.createElement("div");
                    container.classList.add("image-container");
                    container.setAttribute("data-section-no", image.sectionNo);

                    const img = document.createElement("img");
                    img.src = image.thumbnailUrl;
                    
                    img.onerror = function () {
                        img.style.display = "none"; // Hides the broken image icon
                    };

                    const label = document.createElement("span");
                    label.classList.add("image-id");
                    label.textContent = image.sectionNo;
                    label.style.color = sectionColor(image.sectionstatus);

                    if (image.is_annotation === 1) {
                        const greenDot = document.createElement("div");
                        greenDot.classList.add("green-dot");
                        container.appendChild(greenDot);
                    }

                    container.appendChild(img);
                    container.appendChild(label);

                    return container;
                }

                function sectionColor(index) {
                switch(index) {
                    case 0: return 'red';
                    case 1: return 'orange';
                    case 2: return '#f003fc';
                    case 3: return 'blue';
                    case 4: return '#2eb02e';
                    default: return 'red';
                }
            }
                document.querySelector(".image-grid-container").addEventListener("click", async (event) => {
                    const imageContainer = event.target.closest(".image-container");
                    if (!imageContainer) return;

                    const previouslySelected = document.querySelector(".image-container.selected");
                    if (previouslySelected) {
                        previouslySelected.classList.remove("selected");
                    }

                    imageContainer.classList.add("selected");

                    const payload = {
                        id: session_id,
                        tool_name: 'thumbnail_viewer',
                        params: {
                            biosample: biosample,
                            section: imageContainer.getAttribute("data-section-no"),
                        },
                    };

                    try {
                        //const response = await fetch("https://llm.humanbrain.in:1062/context", {
                        const response = await fetch("http://dgx3.humanbrain.in:10603/context", {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json",
                            },
                            body: JSON.stringify(payload),
                        });

                        const data = await response.json();
                        console.log("Server Response:", data);

                        window.parent.postMessage(
                        { action_context: payload },
                        "*" // Replace * with specific origin for security, e.g. "https://yourdomain.com"
                        );
                    console.log("sending context to parent", { action_context: payload });
                    } catch (error) {
                        console.error("Error sending data to context API:", error);
                    }
                });

               function renderImages(grid, index) {
                grid.innerHTML = "";

                const columns = calculateColumns();
                const rowHeight = 150;
                const rows = Math.ceil(window.innerHeight / rowHeight);
                const imagesPerPage = columns * rows;

                grid.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;

                const sortedImages = [...imagesData].sort(
                    (a, b) => parseInt(a.sectionNo) - parseInt(b.sectionNo)
                );
                const paginatedImages = sortedImages.slice(index, index + imagesPerPage);

                let columnWiseImages = Array.from({ length: rows }, () => Array(columns).fill(null));

                let imageIdx = 0;
                for (let col = 0; col < columns; col++) {
                    for (let row = 0; row < rows; row++) {
                        if (imageIdx < paginatedImages.length) {
                            columnWiseImages[row][col] = paginatedImages[imageIdx];
                            imageIdx++;
                        }
                    }
                }

                columnWiseImages.flat().forEach((image) => {
                    if (image) {
                        grid.appendChild(createImageElement(image));
                    }
                });
                }

                function scrollImages(direction) {
                    if (isAnimating) return;
                    isAnimating = true;

                    const columns = calculateColumns();
                    const rows = Math.ceil(window.innerHeight / 150);
                    const imagesPerPage = columns * rows;
                    let nextIndex = startIndex;

                    if (direction === "next") {
                        nextIndex += imagesPerPage;
                    } else {
                        nextIndex -= imagesPerPage;
                    }

                    // Ensure nextIndex stays within bounds
                    nextIndex = Math.max(0, Math.min(nextIndex, imagesData.length - imagesPerPage));

                    if (nextIndex === startIndex) {
                        isAnimating = false;
                        return;
                    }

                    const newGrid = startIndex < nextIndex ? grid2 : grid1;
                    renderImages(newGrid, nextIndex);

                    gridWrapper.style.transition = "transform 0.5s ease-in-out";
                    gridWrapper.style.transform = `translateX(${startIndex < nextIndex ? "-100%" : "100%"})`;

                    setTimeout(() => {
                        gridWrapper.style.transition = "none";
                        gridWrapper.style.transform = "translateX(0)";

                        if (startIndex < nextIndex) {
                            grid1.innerHTML = grid2.innerHTML;
                        } else {
                            grid2.innerHTML = grid1.innerHTML;
                        }

                        startIndex = nextIndex;
                        isAnimating = false;
                        updatePaginationButtons();
                    }, 500);
                }


                function updatePaginationButtons() {
                    const columns = calculateColumns();
                    const rows = Math.ceil(window.innerHeight / 150);
                    const imagesPerPage = columns * rows;

                    prevButton.disabled = startIndex === 0;
                    nextButton.disabled = startIndex + imagesPerPage >= imagesData.length;
                }



                prevButton.addEventListener("click", () => scrollImages("prev"));
                nextButton.addEventListener("click", () => scrollImages("next"));
                window.addEventListener("resize", () => {
                const columns = calculateColumns();
                const rows = Math.ceil(window.innerHeight / 150);
                const imagesPerPage = columns * rows;

                const sortedImages = [...imagesData].sort(
                    (a, b) => parseInt(a.sectionNo) - parseInt(b.sectionNo)
                );

                const maxStart = Math.max(0, sortedImages.length - imagesPerPage);
                if (startIndex > maxStart) {
                    startIndex = maxStart;
                }

                renderImages(grid1, startIndex);
                updatePaginationButtons();
            });

                fetchImages();
            </script>

        </body>
        </html>
            """    

    if biosample not in thumbnail_cache.keys():
        thumbnail_cache[biosample] = {}
        if series_type not in thumbnail_cache[biosample].keys():
            thumbnail_cache[biosample][series_type] = template
    return template

# ==================== Flask Routes ====================

@app.route('/')
def index():
    # Read and return the combined template
    try:
        with open('/home/users/imran/brain_assistant/local/templates/index.html', 'r') as file:
            template = file.read()
        return template
    except FileNotFoundError:
        return '''
        <h1>Combined Brain Viewer</h1>
        <p>Error: index.html template not found!</p>
        <ul>
            <li><a href="/light_weight_viewer">Light Weight Viewer</a></li>
            <li><a href="/thumbnail_viewer">Thumbnail Viewer</a></li>
        </ul>
        '''

@app.route('/combined/<int:biosample>/<int:section>')
@app.route('/combined/<int:biosample>/<int:section>/<series_type>')
def combined_viewer(biosample=222, section=1000, series_type='NISSL'):
    try:
        with open('/home/users/imran/brain_assistant/local/templates/index.html', 'r') as file:
            template = file.read()
        
        # Replace dynamic values in the template
        template = template.replace('currentBiosample = 222', f'currentBiosample = {biosample}')
        template = template.replace('currentSection = 1000', f'currentSection = {section}')
        template = template.replace("currentSeriesType = 'NISSL'", f"currentSeriesType = '{series_type}'")
        template = template.replace('id="atlas-biosample">222', f'id="atlas-biosample">{biosample}')
        template = template.replace('id="atlas-section">1000', f'id="atlas-section">{section}')
        template = template.replace('id="thumbnail-biosample">222', f'id="thumbnail-biosample">{biosample}')
        
        return template
    except FileNotFoundError:
        return f'''
        <h1>Error: Template not found</h1>
        <p>Looking for biosample {biosample}, section {section}, series {series_type}</p>
        '''

@app.route('/light_weight_viewer', methods=['GET', 'POST'])
def light_weight_viewer_route():
    if request.method == 'POST':
        data = request.get_json()
        biosample = data.get('biosample')
        section = data.get('section')
        session_id = data.get('session_id', 'default')
        template = get_light_weight_viewer_temp(biosample, section, session_id=session_id)
        return render_template_string(template)
    
    # GET request - show a form
    return '''
    <h2>Light Weight Viewer</h2>
    <form method="POST" action="/light_weight_viewer">
        <label>Biosample: <input type="number" name="biosample" required></label><br>
        <label>Section: <input type="number" name="section" required></label><br>
        <button type="submit">View</button>
    </form>
    <script>
        document.querySelector('form').onsubmit = function(e) {
            e.preventDefault();
            fetch('/light_weight_viewer', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    biosample: parseInt(document.querySelector('[name=biosample]').value),
                    section: parseInt(document.querySelector('[name=section]').value),
                    session_id: 'test-session'
                })
            })
            .then(r => r.text())
            .then(html => document.body.innerHTML = html);
        }
    </script>
    '''

@app.route('/light_weight_viewer/<int:biosample>/<int:section>')
def light_weight_viewer_direct(biosample, section):
    session_id = request.args.get('session_id', 'default')
    template = get_light_weight_viewer_temp(biosample, section, session_id=session_id)
    return render_template_string(template)

@app.route('/thumbnail_viewer', methods=['GET', 'POST'])
def thumbnail_viewer_route():
    if request.method == 'POST':
        data = request.get_json()
        biosample = data.get('biosample')
        series_type = data.get('series_type', 'NISL')
        session_id = data.get('session_id', 'default')
        roi_section = data.get('roi_section', 1)
        template = get_thumbnail_viewer(biosample, series_type, session_id, roi_section)
        return render_template_string(template)
    
    # GET request - show a form
    return '''
    <h2>Thumbnail Viewer</h2>
    <form method="POST" action="/thumbnail_viewer">
        <label>Biosample: <input type="number" name="biosample" required></label><br>
        <label>Series Type: <input type="text" name="series_type" value="NISL" required></label><br>
        <label>ROI Section: <input type="number" name="roi_section" value="1"></label><br>
        <button type="submit">View</button>
    </form>
    <script>
        document.querySelector('form').onsubmit = function(e) {
            e.preventDefault();
            fetch('/thumbnail_viewer', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    biosample: parseInt(document.querySelector('[name=biosample]').value),
                    series_type: document.querySelector('[name=series_type]').value,
                    roi_section: parseInt(document.querySelector('[name=roi_section]').value),
                    session_id: 'test-session'
                })
            })
            .then(r => r.text())
            .then(html => document.body.innerHTML = html);
        }
    </script>
    '''

@app.route('/thumbnail_viewer/<int:biosample>/<series_type>/<session_id>/<int:roi_section>')
def thumbnail_viewer_direct(biosample, series_type, session_id, roi_section):
    template = get_thumbnail_viewer(biosample, series_type, session_id, roi_section)
    return render_template_string(template)

# API endpoints for context
@app.route('/context', methods=['POST'])
def context():
    data = request.get_json()
    # Process context data as needed
    return jsonify({"status": "success", "data": data})

# Legacy compatibility endpoints
@app.route('/get_light_weight_viewer', methods=['POST'])
def get_light_weight_viewer_api():
    data = request.get_json()
    user_query = data.get('query')
    result = get_light_weight_viewer(user_query)
    return jsonify({"template": result})

@app.route('/get_thumbnail_viewer', methods=['POST'])
def get_thumbnail_viewer_api():
    data = request.get_json()
    query = data.get('query')
    session_id = data.get('session_id', 'default')
    chat_history = data.get('chat_history')
    page_context = data.get('page_context')
    result = thumbnail_viewer(query, session_id, chat_history, page_context)
    return jsonify({"url": result})

# ==================== Chatbot API Endpoints ====================

@app.route('/api/brain-region', methods=['POST'])
def process_brain_region():
    """Process brain region queries"""
    try:
        data = request.json
        region_name = data.get('region')
        mode = data.get('mode', 'fast')  # fast, web, or ultra
        
        # Input validation
        if not isinstance(region_name, str) or len(region_name.strip()) == 0:
            return jsonify({
                'success': False,
                'message': 'Please provide a valid brain region name'
            }), 400
        
        region_name = region_name.strip()
        
        # Validate mode
        if mode not in ['fast', 'web', 'ultra']:
            mode = 'fast'
        
        if not region_name:
            return jsonify({
                'success': False,
                'message': 'Please provide a brain region name'
            }), 400
        
        # Sync conversation history before processing
        sync_assistant_history()
        
        # Add user query to chat history
        add_to_chat_history('user_query', f"Tell me about {region_name} (mode: {mode})", region_name)
        
        # Get brain region info
        is_valid, info = assistant.get_brain_region_info(region_name, mode)
        
        if is_valid:
            # Store current region and mode for follow-up questions
            assistant.current_region = region_name
            session['current_mode'] = mode
            session.modified = True
            
            # Add assistant response to chat history
            add_to_chat_history('region_info', info, region_name)
            
            return jsonify({
                'success': True,
                'message': info,
                'region': region_name,
                'mode': mode,
                'chat_history': get_chat_history()
            })
        else:
            # Add error response to chat history
            add_to_chat_history('assistant_response', info, region_name)
            
            return jsonify({
                'success': False,
                'message': info,
                'chat_history': get_chat_history()
            })
            
    except Exception as e:
        error_msg = f'Error: {str(e)}'
        add_to_chat_history('assistant_response', error_msg)
        return jsonify({
            'success': False,
            'message': error_msg,
            'chat_history': get_chat_history()
        }), 500

@app.route('/api/ask-question', methods=['POST'])
def ask_question():
    """Handle follow-up questions about current brain region"""
    try:
        data = request.json
        question = data.get('question')
        use_web = data.get('use_web', False)
        # Use stored mode from session, fallback to provided mode or 'fast'
        mode = session.get('current_mode', data.get('mode', 'fast'))
        
        # Input validation
        if not isinstance(question, str) or len(question.strip()) == 0:
            return jsonify({
                'success': False,
                'message': 'Please provide a valid question'
            }), 400
        
        question = question.strip()
        
        if not question:
            return jsonify({
                'success': False,
                'message': 'Please provide a question'
            }), 400
        
        if not assistant.current_region:
            return jsonify({
                'success': False,
                'message': 'Please select a brain region first'
            }), 400
        
        # Sync conversation history before processing
        sync_assistant_history()
        
        # Automatically enable web search if mode is 'web' (Detailed mode)
        if mode == 'web':
            use_web = True
        
        # Add user question to chat history
        search_indicator = " (web search)" if use_web else ""
        add_to_chat_history('user_query', f"{question}{search_indicator}", assistant.current_region)
        
        # Get answer
        answer = assistant.ask_question(question, use_web)
        
        # Add assistant answer to chat history
        add_to_chat_history('assistant_response', answer, assistant.current_region)
        
        return jsonify({
            'success': True,
            'message': answer,
            'region': assistant.current_region,
            'chat_history': get_chat_history(),
            'web_search_used': use_web
        })
        
    except Exception as e:
        error_msg = f'Error: {str(e)}'
        add_to_chat_history('assistant_response', error_msg, assistant.current_region)
        return jsonify({
            'success': False,
            'message': error_msg,
            'chat_history': get_chat_history()
        }), 500

@app.route('/api/ask-question-stream', methods=['POST'])
def ask_question_stream():
    """Handle follow-up questions with streaming response"""
    from flask import Response, stream_with_context
    import json
    
    data = request.json
    question = data.get('question')
    use_web = data.get('use_web', False)
    mode = session.get('current_mode', data.get('mode', 'fast'))
    
    if not question:
        return jsonify({'success': False, 'message': 'Please provide a question'}), 400
    
    if not assistant.current_region:
        return jsonify({'success': False, 'message': 'Please select a brain region first'}), 400
    
    # Sync conversation history before processing
    sync_assistant_history()
    
    # Automatically enable web search if mode is 'web' (Detailed mode)
    if mode == 'web':
        use_web = True
    
    # Add user question to chat history
    search_indicator = " (web search)" if use_web else ""
    add_to_chat_history('user_query', f"{question}{search_indicator}", assistant.current_region)
    
    def generate():
        try:
            # Get streaming answer
            full_response = ""
            for chunk in assistant.ask_question_stream(question, use_web):
                full_response += chunk
                # Send each chunk as Server-Sent Event
                yield f"data: {json.dumps({'chunk': chunk, 'success': True})}\n\n"
            
            # Add complete answer to chat history
            add_to_chat_history('assistant_response', full_response, assistant.current_region)
            
            # Send final message with complete status
            yield f"data: {json.dumps({'complete': True, 'success': True, 'region': assistant.current_region, 'web_search_used': use_web})}\n\n"
            
        except Exception as e:
            error_msg = f'Error: {str(e)}'
            add_to_chat_history('assistant_response', error_msg, assistant.current_region)
            yield f"data: {json.dumps({'error': error_msg, 'success': False})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

@app.route('/api/brain-region-stream', methods=['POST'])
def process_brain_region_stream():
    """Process brain region queries with streaming response"""
    from flask import Response, stream_with_context
    import json
    
    data = request.json
    region_name = data.get('region')
    mode = data.get('mode', 'fast')
    
    if not region_name:
        return jsonify({'success': False, 'message': 'Please provide a brain region name'}), 400
    
    # Sync conversation history before processing
    sync_assistant_history()
    
    # Add user query to chat history
    add_to_chat_history('user_query', f"Tell me about {region_name} (mode: {mode})", region_name)
    
    def generate():
        try:
            # First validate if it's a brain region
            is_valid = assistant.validate_brain_region(region_name)
            
            if not is_valid:
                error_msg = f"'{region_name}' is not a brain region. Please enter a valid brain region name."
                yield f"data: {json.dumps({'chunk': error_msg, 'success': False})}\n\n"
                yield f"data: {json.dumps({'complete': True, 'success': False})}\n\n"
                return
            
            # Store current region and mode
            assistant.current_region = region_name
            session['current_mode'] = mode
            session.modified = True
            
            # Get streaming info
            full_response = ""
            for chunk in assistant.get_brain_region_info_stream(region_name, mode):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk, 'success': True})}\n\n"
            
            # Add complete response to chat history
            add_to_chat_history('region_info', full_response, region_name)
            
            # Send final message
            yield f"data: {json.dumps({'complete': True, 'success': True, 'region': region_name, 'mode': mode})}\n\n"
            
        except Exception as e:
            error_msg = f'Error: {str(e)}'
            add_to_chat_history('assistant_response', error_msg, region_name)
            yield f"data: {json.dumps({'error': error_msg, 'success': False})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

@app.route('/api/validate-region', methods=['POST'])
def validate_region():
    """Validate if input is a brain region"""
    try:
        data = request.json
        region_name = data.get('region')
        
        # Input validation
        if not isinstance(region_name, str) or len(region_name.strip()) == 0:
            return jsonify({
                'success': False,
                'is_valid': False,
                'message': 'Please provide a valid region name'
            }), 400
        
        region_name = region_name.strip()
        
        if not region_name:
            return jsonify({
                'success': False,
                'is_valid': False
            }), 400
        
        is_valid = assistant.validate_brain_region(region_name)
        
        return jsonify({
            'success': True,
            'is_valid': is_valid
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/chat-history', methods=['GET'])
def get_chat_history_endpoint():
    """Get current chat history"""
    try:
        chat_history = get_chat_history()
        return jsonify({
            'success': True,
            'chat_history': chat_history,
            'current_region': assistant.current_region,
            'current_mode': session.get('current_mode', 'fast')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/chat-history', methods=['DELETE'])
def clear_chat_history_endpoint():
    """Clear chat history"""
    try:
        clear_chat_history()
        # Also reset current region
        assistant.current_region = None
        return jsonify({
            'success': True,
            'message': 'Chat history cleared successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/search-config', methods=['POST'])
def configure_search():
    """Configure search parameters"""
    try:
        data = request.json
        max_sources = data.get('max_sources', 12)
        
        if max_sources < 1 or max_sources > 20:
            return jsonify({
                'success': False,
                'message': 'max_sources must be between 1 and 20'
            }), 400
        
        # Update search configuration
        assistant.set_search_sources(max_sources)
        
        return jsonify({
            'success': True,
            'message': f'Search configured to use maximum {max_sources} sources',
            'max_sources': max_sources
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/performance-stats', methods=['GET'])
def get_performance_stats():
    """Get performance statistics"""
    try:
        # Get stats if available in improved version
        if hasattr(assistant, 'get_performance_stats'):
            stats = assistant.get_performance_stats()
            return jsonify({
                'success': True,
                'stats': stats
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Performance stats not available in current version'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

if __name__ == '__main__':
    import sys
    port = 5001
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('--port='):
                port = int(arg.split('=')[1])
    app.run(debug=True, host='0.0.0.0', port=port)