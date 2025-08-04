import requests
import pickle
import sys
from IPython.display import HTML
from  openai import OpenAI
import ast
import asyncio
from sqlalchemy import create_engine, text
from langchain.sql_database import SQLDatabase
sys.path.append("/apps/src/utils/agent/tools/actions/")
from mini_atlas.helper import get_geojson
from urls import base_url
from pathlib import Path

# with open("json_file_name_dict.pkl", "rb") as f:
#     json_file_name_dict = pickle.load(f)


MySQL_db_user = "root"
MySQL_db_password = "Health#123"
# MySQL_db_host = "dev2mani.humanbrain.in"
MySQL_db_host = "apollo2.humanbrain.in"
MySQL_db_port = "3306"
MySQL_db_name = "HBA_V2"
MySQL_db = SQLDatabase.from_uri(f"mysql+pymysql://{MySQL_db_user}:{MySQL_db_password}@{MySQL_db_host}:{MySQL_db_port}/{MySQL_db_name}")

MySQL_DATABASE_URL = f"mysql+pymysql://{MySQL_db_user}:{MySQL_db_password}@{MySQL_db_host}:{MySQL_db_port}/{MySQL_db_name}"
# MySQL_engine = create_engine(MySQL_DATABASE_URL)
MySQL_engine = create_engine(
    MySQL_DATABASE_URL,
    pool_pre_ping=True,    # Check if connection is alive before using
    pool_recycle=21600      # Recycle connections every hour
)

geojson_cache = {}
light_weight_viewer_cache = {}

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
        # print("Response from api ____",response.json()['msg'][0]['status'])
        if response.status_code != 200:
            return "Failed"
        # elif response.status_code == 200 and response.json()['msg'][0]['status'] == False :
        elif response.status_code == 200 and response.json()['status'] == False :
            return "Biosample or Section Not Found!"

        data = response.json()['msg'][0]

        # area_and_perimeter_calculation(data)
        if biosample not in geojson_cache.keys():
            geojson_cache[biosample] = {}
        geojson_cache[biosample][section] = data

        # print(data)
        return data

def MySQL_db_retriever(sql_query):
    with MySQL_engine.connect() as connection:  
        result = connection.execute(text(sql_query))  
    data = result.fetchall()
    return data

def get_metadata(biosample, section):
    #sereies type = 1 is NISL
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
    
    #if the export status from section table is 4 then the image can be loaded in tif
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
        # tools=tools,
        temperature=0,
    )

    return completion.choices[0].message.content

with open("/apps/src/utils/agent/tools/actions/light_weight_viewer/jp2_file_names.pkl", "rb") as f:
    jp2_file_names = pickle.load(f)

with open("/apps/src/utils/agent/tools/actions/light_weight_viewer/json_file_name_dict.pkl", "rb") as g:
    json_file_name_dict = pickle.load(g)


def get_json_filename(biosample, section, ontology):
    path = F"/apps/analytics/{biosample}/appData/atlasEditor/{ontology}/NISL/{section}/"
    file_name = None
    for p in Path(path).rglob("*.json"):
        if "FlatTree" in p.parts[-1]:
            print(p.parts)
            file_name = p.parts[-1]
    return file_name
    
def imageInfo(biosample,section,jp2_file_name):
    # jp2_file_name = jp2_file_names[str(biosample)][str(section)]
    # imgUrl = f"http://dev2mani.humanbrain.in:9081/fcgi-bin/iipsrv.fcgi?FIF=/data/storageIIT/humanbrain/analytics/{biosample}/NISL/"+ jp2_file_name +"&WID=1024&GAM=1.4&MINMAX=1:0,255&MINMAX=2:0,255&MINMAX=3:0,255&JTL={z},{tileIndex}"
    imgUrl = f"https://apollo2.humanbrain.in/iipsrv/fcgi-bin/iipsrv.fcgi?FIF=/ddn/storageIIT/humanbrain/analytics/{biosample}/NISL/"+ jp2_file_name +"&WID=1024&GAM=1.4&MINMAX=1:0,255&MINMAX=2:0,255&MINMAX=3:0,255&JTL={z},{tileIndex}"
    # imgUrl = f"http://llm.humanbrain.in:9081/fcgi-bin/iipsrv.fcgi?FIF=/ddn/storageIIT/humanbrain/analytics/{biosample}/NISL/"+ jp2_file_name +"&WID=1024&GAM=1.4&MINMAX=1:0,255&MINMAX=2:0,255&MINMAX=3:0,255&JTL={z},{tileIndex}"
    print(imgUrl)
    return imgUrl

def geojsonUrl(biosample,section,ontology):
    # json_file_name = json_file_name_dict[str(biosample)][str(section)]
    json_file_name = get_json_filename(biosample, section, ontology)
    if json_file_name is None:
            json_file_name = ""
    jsonUrl  = f"https://apollo2.humanbrain.in/iipsrv/ddn/storageIIT/humanbrain/analytics/{biosample}/appData/atlasEditor/{ontology}/NISL/{section}/"+ json_file_name
    return jsonUrl

def get_light_weight_viewer_temp(biosample, section, stain_id=None, session_id=None):
    # resp = llm_resp(user_query)
    # resp = ast.literal_eval(resp)
    # biosample = resp['biosample']
    # section  = resp['section']
    # data = get_geojson(biosample, section)

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

    # geojson = data['geoJson']
    geourl = geojsonUrl(biosample,section, ontology)
    imgurl = imageInfo(biosample, section, jp2_file_name)
    # user_id = data['userId']
    print("INSIDE LIGHT WEIGHT VIEWER")
    print("IMAGE URL:", imgurl)
    print("GEO URL:", geourl)
    print("SESSION ID:", session_id)

    if biosample in light_weight_viewer_cache.keys():
        if section in light_weight_viewer_cache[biosample].keys():
            print("Rendering light weight viewer from cache")
            template = light_weight_viewer_cache[biosample][section]
    else:   

        # template = """<!doctype html>
        #         <html lang="en" style="height:100%;">
        #         <head>
        #             <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/openlayers/openlayers.github.io@master/en/v5.3.0/css/ol.css" type="text/css">
        #             <style>
        #                 html, body {
        #                     margin: 0;
        #                     padding: 0;
        #                     overflow: hidden;
        #                 }
        #                 .header{
        #                     margin: 0;
        #                     padding-left: 15px;
        #                     color: black;
        #                     font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        #                 }

        #                 .header p{
        #                     margin: 0;
        #                     padding-top: 5px;
        #                     font-size: 12px;
        #                     font-weight: bold;
        #                 }
        #                 .name-container{
        #                     display: flex;
        #                     width: 100%;
        #                     height: 0;
        #                 }
        #                 .feature-name{
        #                     font-size: 20px;
        #                     padding: 10px 40px;
        #                     z-index: 3;
        #                 }
        #                 .map {
        #                 height:100%;
        #                 width:100%;
        #                 }
                        
        #                 /* Opacity slider styles */
        #                 .slider-container {
        #                     position: absolute;
        #                     left: 15px;
        #                     top: 55%;
        #                     transform: translateY(-50%);
        #                     height: 120px;
        #                     width: 20px;
        #                     background-color: rgba(255, 255, 255, 0.7);
        #                     border-radius: 8px;
        #                     padding: 10px 0;
        #                     display: flex;
        #                     flex-direction: column;
        #                     align-items: center;
        #                     box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        #                     z-index: 1000;
        #                 }
                        
        #                 #opacity-slider {
        #                     writing-mode: bt-lr;
        #                     -webkit-appearance: slider-vertical;
        #                     width: 3px;
        #                     height: 100px;
        #                     padding: 0;
        #                     margin: 0;
        #                 }
                        
        #                 /* Make slider thumb (handle) smaller */
        #                 #opacity-slider::-webkit-slider-thumb {
        #                     transform: scale(0.6);
        #                     cursor: pointer;
        #                 }
                        
        #                 /* For Firefox */
        #                 #opacity-slider::-moz-range-thumb {
        #                     transform: scale(0.6);
        #                     cursor: pointer;
        #                 }
                        
        #                 /* For IE/Edge */
        #                 #opacity-slider::-ms-thumb {
        #                     transform: scale(0.6);
        #                     cursor: pointer;
        #                 }
                        
        #                 .opacity-value {
        #                     font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        #                     font-size: 10px;
        #                     margin-top: 5px;
        #                 }
        #             </style>
        #             <script src="https://cdn.jsdelivr.net/gh/openlayers/openlayers.github.io@master/en/v5.3.0/build/ol.js"></script>
        #             <title>Atlas Viewer</title>
        #             </head>
        #             <body style="height:100%;">
        #                     <div class="header">"""+f"""
        #                         <p>Atlas Viewer - Biosample: {biosample} Section:{section}</p>"""+"""
        #                     </div>
        #                     <div class="name-container">
        #                         <span class="feature-name" id="feature-name"></span>
        #                     </div>
        #                     <div id="map" class="map"></div>
                            
        #                     <!-- Opacity slider control -->
        #                     <div class="slider-container">
        #                         <input type="range" min="0" max="99" value="50" class="slider" id="opacity-slider" orient="vertical">
        #                         <div class="opacity-value" id="opacity-value">50</div>
        #                     </div>
                            
        #                     <script type="text/javascript">

        #                     const Feature = ol.Feature;
        #                     const Map = ol.Map;
        #                     const View = ol.View;
        #                     const GeoJSON = ol.format.GeoJSON;
        #                     const Circle = ol.geom.Circle;
        #                     const Point = ol.geom.Point;
        #                     const Icon = ol.style.Icon;
        #                     const TileLayer = ol.layer.Tile;
        #                     const VectorLayer = ol.layer.Vector;
        #                     const OSM = ol.source.OSM;
        #                     const VectorSource = ol.source.Vector;
        #                     const CircleStyle = ol.style.Circle;
        #                     const Fill = ol.style.Fill;
        #                     const Stroke = ol.style.Stroke;
        #                     const Style = ol.style.Style;
        #                     const Zoomify = ol.source.Zoomify;
        #                     const Select = ol.interaction.Select;
        #                     """ + f"""

        #                     const geojsonUrl = '{geourl}'
        #                     const mapurl = '{imgurl}'
        #                     const mapsiz = [{width},{height}];
        #                     const rigidrotation = {rotation};
        #                     const biosample = '{biosample}';
        #                     const section = '{section}';
        #                     const tool_name= 'mini_atlas';
        #                     const session_id = '{session_id}';
        #                     const parsed_session_id = session_id;
        #                     const trs_rot = {trs_rot};
        #                     """ + """
                            
        #                     // Global variable for opacity
        #                     let opacityAtlas = '50';
        #                     let vectorSource;

        #                     window.addEventListener('message', (event) => {
        #                     // You can check event.origin here for security
        #                     console.log('Message received in iframe:', event.data);

        #                     if (event.data.action === 'sayHi') {
        #                         alert('Got this from parent: ' + event.data.data); // shows "hi"
        #                     }
        #                     });


        #                     async function getGeoJSON(url) {
        #                         try {
        #                             const response = await fetch(url);
        #                             return await response.json();
        #                         } catch (error) {
        #                             console.error('Error fetching GeoJSON:', error);
        #                             return null;
        #                         }
        #                     }

        #                     function styleFunction(feature) {
        #                         if(typeof(opacityAtlas) == "number") {    
        #                         opacityAtlas = '50'
        #                         }
        #                         var data = feature.get('data')
        #                         var clr = '#FF000070'

        #                         if (data !== undefined && data.color_hex_triplet != undefined){
        #                             clr =  data.color_hex_triplet;
        #                             if(!data.color_hex_triplet.startsWith('#')){
        #                             clr= '#'+clr
        #                             }
        #                         }

        #                         if(clr.length==7) {
        #                             var atlasClr = clr;
        #                             clr = clr + opacityAtlas;
        #                         }

        #                         var st = [
        #                             new Style({
        #                             zIndex: -1,
        #                             stroke: new Stroke({
        #                                 color: atlasClr,
        #                                 width: 2.5,
        #                             }),
        #                             fill: new Fill({
        #                                 color: clr
        #                             }),
        #                             }),
        #                         ];

        #                         var geometry = feature.getGeometry();
        #                         if (geometry.getType() === 'LineString') {
        #                             var coordinates = geometry.getCoordinates();
        #                             if (coordinates.length >= 2) {
        #                             var start = coordinates[coordinates.length - 2];
        #                             var end = coordinates[coordinates.length - 1];
                                    
        #                             var dx = end[0] - start[0];
        #                             var dy = end[1] - start[1];
        #                             var rotation = Math.atan2(dy, dx);

        #                             st.push(new Style({
        #                                     stroke: new Stroke({
        #                                     color: '#000000',
        #                                     width: 3,
        #                                     })
        #                                 }));
                                    
        #                             st.push(new Style({
        #                                 geometry: new Point(end),
        #                                 image: new Icon({
        #                                 src: 'https://apollo2.humanbrain.in/viewer/assets/images/colorsvg/right_arrow.svg',
        #                                 anchor: [0.75, 0.5],
        #                                 rotateWithView: true,
        #                                 rotation: -rotation,
        #                                 })
        #                             }));
        #                             } 
        #                         }
                                
        #                         st.push(new Style({
        #                             zIndex: -1,
        #                             image: new CircleStyle({
        #                             radius: 5.9,
        #                             stroke: new Stroke({
        #                                 color: atlasClr,
        #                                 width: 1,
        #                             }),
        #                             fill: new Fill({
        #                                 color: clr,
        #                             }),
        #                             }),
        #                             geometry: function (feature) {
        #                             var coordinates = feature.getGeometry().getCoordinates();
        #                             return new Point(coordinates);
        #                             },
        #                         }));

        #                         return st;
        #                     }
                            
        #                     // Update all features with the new opacity
        #                     function updateLayerOpacity() {
        #                         if (vectorSource) {
        #                             vectorSource.getFeatures().forEach(feature => {
        #                                 feature.setStyle(styleFunction(feature));
        #                             });
        #                         }
        #                     }

        #                     // Call the function to get data
        #                     getGeoJSON(geojsonUrl).then(geojsonData => {
        #                         const geojsonObject = geojsonData;

        #                         const vectorLayer = new VectorLayer({
        #                             transition: 0,
        #                             source: new VectorSource({
        #                                 format: new GeoJSON(),
        #                                 wrapX: false,
        #                             }),
        #                             style: styleFunction,
        #                         });

        #                         const zoomifySource = new Zoomify({
        #                             url: mapurl,
        #                             size: mapsiz,
        #                             crossOrigin: "anonymous",
        #                             tierSizeCalculation: 'truncated',
        #                             imageSmoothing: false,
        #                             tileSize: 2048
        #                         });

        #                         const imagery = new TileLayer({
        #                             source: zoomifySource
        #                         });

        #                         const extent = zoomifySource.getTileGrid().getExtent();

        #                         const map = new Map({
        #                             layers: [imagery, vectorLayer],
        #                             target: 'map',
        #                             view: new View({
        #                                 zoom: 10,
        #                                 minZoom: 8, 
        #                                 maxZoom: 19,
        #                                 rotation: (rigidrotation * Math.PI / 180),
        #                                 extent: extent
        #                             }),
        #                             controls: ol.control.defaults({
        #                                 rotate: false 
        #                             })
        #                         });

        #                         map.getView().fit(imagery.getSource().getTileGrid().getExtent());
        #                         if(trs_rot != 0)
        #                             map.getView().setRotation(trs_rot);
        #                         var centerMap = map.getView().getCenter();
        #                         vectorSource = vectorLayer.getSource();

        #                         var features = vectorSource.getFormat().readFeatures(geojsonObject);
        #                         features.forEach(element => {
        #                             var elementRotate = element.getGeometry();
        #                             var xy = centerMap;
        #                             elementRotate = elementRotate.rotate((( rigidrotation) * Math.PI / 180), xy);
        #                         });
        #                         vectorSource.addFeatures(features);
        #                         vectorSource.getFeatures().forEach(element => {
        #                             element.setStyle(styleFunction(element));
        #                         });

                                
        #                         const selectInteraction = new Select({
        #                             condition: ol.events.condition.singleClick,
        #                             style: null, // Disable default OpenLayers selection style
        #                         });

        #                         map.addInteraction(selectInteraction);

        #                         selectInteraction.on('select', async function (e) {
        #                             // Reset all styles
        #                             vectorSource.getFeatures().forEach(feature => {
        #                                 feature.setStyle(styleFunction(feature));  
        #                             });

        #                             const selected = e.selected;
        #                             const featureNameEl = document.getElementById("feature-name");

        #                             if (selected.length > 0) {
        #                                 const clickedFeature = selected[0].getProperties();
        #                                 const clickedId = clickedFeature.data.id;
        #                                 const clickedName = clickedFeature.data.name;

        #                                 // Highlight all features with the same ID
        #                                 const matchingFeatures = vectorSource.getFeatures().filter(f => {
        #                                     const data = f.get('data');
        #                                     return data && data.id === clickedId;
        #                                 });

        #                                 matchingFeatures.forEach(f => {
        #                                     f.setStyle(new Style({
        #                                         stroke: new Stroke({
        #                                             color: 'red',
        #                                             width: 3
        #                                         }),
        #                                         fill: new Fill({
        #                                             color: 'transparent'
        #                                         })
        #                                     }));
        #                                 });

        #                                 featureNameEl.textContent = ` ${clickedName}`;

        #                                 const payload = {
        #                                     id: parsed_session_id,
        #                                     tool_name: tool_name,
        #                                     params: {
        #                                         id: clickedId,
        #                                         name: clickedName,
        #                                         biosample: biosample,
        #                                         section: section
        #                                     }
        #                                 };
        #                                 window.parent.postMessage(payload, '*');

        #                                 try {
        #                                     const response = await fetch("https://llm.humanbrain.in:1062/context", {
        #                                         method: "POST",
        #                                         headers: {
        #                                             "Content-Type": "application/json"
        #                                         },
        #                                         body: JSON.stringify(payload)
        #                                     });
        #                                     const data = await response.json();
        #                                     console.log("Server Response:", data);
        #                                 } catch (error) {
        #                                     console.error("Error sending data to context API:", error);
        #                                 }

        #                             } else {
        #                                 // Clicked outside any feature
        #                                 featureNameEl.textContent = "";

        #                                 const payload = {
        #                                     id: parsed_session_id,
        #                                     tool_name: tool_name,
        #                                     params: {
        #                                         id: -1,
        #                                         name: null,
        #                                         biosample: biosample,
        #                                         section: section
        #                                     }
        #                                 };
        #                                 window.parent.postMessage(payload, '*');
        #                             }
        #                         });

                                
        #                         // Set up opacity slider interaction
        #                         let lastOpacity = opacityAtlas;
        #                         let throttleTimeout = null;

        #                         const opacitySlider = document.getElementById("opacity-slider");
        #                         const opacityValueDisplay = document.getElementById("opacity-value");

        #                         function throttledOpacityUpdate(newOpacity) {
        #                             if (newOpacity === lastOpacity) return;
        #                             lastOpacity = newOpacity;

        #                             opacityAtlas = newOpacity;
        #                             opacityValueDisplay.textContent = opacityAtlas;
        #                             updateLayerOpacity();

        #                             if(selectInteraction){
        #                                 selectInteraction.getFeatures().clear();
        #                             }

        #                             document.getElementById("feature-name").textContent = "";
        #                         }

        #                         opacitySlider.addEventListener("input", function () {
        #                             const newOpacity = this.value.toString().padStart(2, '0');

        #                             if (throttleTimeout) clearTimeout(throttleTimeout);

        #                             throttleTimeout = setTimeout(() => {
        #                                 throttledOpacityUpdate(newOpacity);
        #                             }, 10); 
        #                         });
                            
        #                     });
        #             // Updated message listener to trigger click events
        #                     window.addEventListener('message', (event) => {
        #                         console.log('Message received in iframe:', event.data);

        #                         // Handle the existing 'sayHi' action
        #                         if (event.data.action === 'sayHi') {
        #                             alert('Got this from parent: ' + event.data.data);
        #                         }

        #                         // Handle highlight action with region data
        #                         if (event.data.action === 'highlight' && event.data.region) {
        #                             const { id } = event.data.region;
        #                             console.log(`Triggering click event for region ID: ${id}`);
        #                             triggerClickEventById(id);
        #                         }

        #                         // Handle direct region data (your format: {id: 158, name: 'Ganglionic eminence', acronym: 'GE'})
        #                         if (event.data.id && event.data.name && event.data.type == "HighlightRegion" && typeof event.data.id === 'number') {
        #                             console.log(`Triggering click event for region: ${event.data.name} (ID: ${event.data.id})`);
        #                             triggerClickEventById(event.data.id);
        #                         }
        #                     });
        #                 </script>
        #             </body>
        #         </html>
        #     """
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

    # with open("/apps/src/utils/agent/tools/actions/light_weight_viewer/light_weight_viewer_2.html", "w") as h:
    #     h.write(template)
 
    # return "http://dgx3.humanbrain.in:10605/ol"
    return template
    # return "http://dgx5.humanbrain.in:10607/ol"

    # return "http://dgx3.humanbrain.in:10607/thumbnail/viewer"
    # return "http://dgx1.humanbrain.in:8085/"


# res = get_light_weight_viewer_temp(222,916)
# print(res)

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
    # return "http://dgx1.humanbrain.in:8085/"


# get_light_weight_viewer(222,1000)