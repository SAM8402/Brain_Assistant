// ==================== Configuration ====================
// These values would be dynamically set by the Flask app
const THUMBNAIL_API_URL = `https://llm.humanbrain.in:1062/get/brain/thumbnails/100`;
const GEOJSON_URL = `https://apollo2.humanbrain.in/iipsrv/ddn/storageIIT/humanbrain/analytics/222/appData/atlasEditor/189/NISL/1000/222-NISL-1000-FlatTree::IIT:V1:SS-100:306:1000:1000.json`;
const IMAGE_URL = `https://apollo2.humanbrain.in/iipsrv/fcgi-bin/iipsrv.fcgi?FIF=/ddn/storageIIT/humanbrain/analytics/222/NISL/B_222_FB74-SL_334-ST_NISL-SE_1000_compressed.jp2&WID=1024&GAM=1.4&MINMAX=1:0,255&MINMAX=2:0,255&MINMAX=3:0,255&JTL={z},{tileIndex}`;

// Initial configuration
let currentBiosample = 222;
let currentSection = 1000;
let currentSeriesType = 'NISSL';
const session_id = 'demo_session';
const tool_name = 'combined_viewer';

// ==================== Panel Resize Functionality ====================
let isResizing = false;
const resizeHandle = document.querySelector('.resize-handle');
const atlasPanel = document.querySelector('.atlas-panel');
const thumbnailPanel = document.querySelector('.thumbnail-panel');
const mainContainer = document.querySelector('.main-container');

resizeHandle.addEventListener('mousedown', function(e) {
isResizing = true;
document.body.style.cursor = 'row-resize';
document.body.style.userSelect = 'none';
});

document.addEventListener('mousemove', function(e) {
if (!isResizing) return;

const containerRect = mainContainer.getBoundingClientRect();
const newAtlasHeight = ((e.clientY - containerRect.top) / containerRect.height) * 100;

if (newAtlasHeight >= 20 && newAtlasHeight <= 80) {
    atlasPanel.style.height = newAtlasHeight + '%';
    thumbnailPanel.style.height = (100 - newAtlasHeight) + '%';
    
    if (window.map) {
        setTimeout(() => window.map.updateSize(), 100);
    }
}
});

document.addEventListener('mouseup', function() {
if (isResizing) {
    isResizing = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
}
});

// ==================== Chatbot Resize Functionality ====================
let isChatbotResizing = false;
let chatbotWidth = 400; // Default width
const chatbotColumn = document.getElementById('chatbotColumn');
const chatbotResizeHandle = document.getElementById('chatbotResizeHandle');
const viewersColumn = document.querySelector('.viewers-column');

// Load saved chatbot width from localStorage
const savedWidth = localStorage.getItem('chatbotWidth');
if (savedWidth) {
chatbotWidth = parseInt(savedWidth);
updateChatbotWidth(chatbotWidth);
}

function updateChatbotWidth(width) {
chatbotColumn.style.width = width + 'px';
document.documentElement.style.setProperty('--chatbot-width', width + 'px');
// Save to localStorage
localStorage.setItem('chatbotWidth', width.toString());
}

chatbotResizeHandle.addEventListener('mousedown', function(e) {
e.preventDefault();
isChatbotResizing = true;
document.body.classList.add('chatbot-resizing');
chatbotColumn.classList.add('resizing');
});

document.addEventListener('mousemove', function(e) {
if (!isChatbotResizing) return;

const windowWidth = window.innerWidth;
const newWidth = windowWidth - e.clientX;

// Constrain width between 300px and 800px, and not more than 70% of window width
const minWidth = 300;
const maxWidth = Math.min(800, windowWidth * 0.7);

if (newWidth >= minWidth && newWidth <= maxWidth) {
    chatbotWidth = newWidth;
    updateChatbotWidth(chatbotWidth);
}
});

document.addEventListener('mouseup', function() {
if (isChatbotResizing) {
    isChatbotResizing = false;
    document.body.classList.remove('chatbot-resizing');
    chatbotColumn.classList.remove('resizing');
}
});

// ==================== Atlas Viewer Code ====================
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

// Atlas viewer variables
let opacityAtlas = '50';
let vectorSource;
let selectInteraction;
let annotationsVisible = true;
let currentSelectedId = null;
let vectorLayer;
let map;
let notAvailableTimeout = null;

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

function updateLayerOpacity() {
if (vectorSource) {
    vectorSource.getFeatures().forEach(feature => {
        const data = feature.get('data');
        const isHighlighted = currentSelectedId !== null && data && data.id === currentSelectedId;
        
        if (isHighlighted) {
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
            feature.setStyle(styleFunction(feature));
        }
    });
}
}

function triggerClickEventById(targetId) {
if (!vectorSource || !selectInteraction) {
    console.error('Vector source or select interaction not available');
    return;
}

selectInteraction.getFeatures().clear();

const featureNameEl = document.getElementById("feature-name");

if (targetId === -1) {
    currentSelectedId = null;
    vectorSource.getFeatures().forEach(feature => {
        feature.setStyle(styleFunction(feature));
    });

    if (featureNameEl) {
        featureNameEl.textContent = "";
        featureNameEl.classList.remove('visible');
    }

    console.log('Cleared all feature selections');
    return;
}

const matchingFeatures = vectorSource.getFeatures().filter(f => {
    const data = f.get('data');
    return data && data.id === targetId;
});

if (matchingFeatures.length > 0) {
    currentSelectedId = targetId;
    
    vectorSource.getFeatures().forEach(feature => {
        feature.setStyle(styleFunction(feature));
    });

    selectInteraction.getFeatures().push(matchingFeatures[0]);

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
        f.changed();
    });
    
    vectorSource.changed();
    vectorLayer.changed();
    
    const featureData = matchingFeatures[0].get('data');
    if (featureData && featureData.name && featureNameEl) {
        if (notAvailableTimeout) {
            clearTimeout(notAvailableTimeout);
            notAvailableTimeout = null;
        }
        featureNameEl.textContent = ` ${featureData.name}`;
        featureNameEl.style.color = "";
        featureNameEl.classList.add('visible');
    }
    console.log(`Highlighted ${matchingFeatures.length} features with ID: ${targetId}`);
} else {
    currentSelectedId = null;
    
    if (selectInteraction) {
        selectInteraction.getFeatures().clear();
    }
    
    vectorSource.getFeatures().forEach(feature => {
        feature.setStyle(styleFunction(feature));
    });
    
    if (featureNameEl) {
        if (notAvailableTimeout) {
            clearTimeout(notAvailableTimeout);
        }
        
        featureNameEl.textContent = "Region not available in this section";
        featureNameEl.style.color = "#888";
        featureNameEl.classList.add('visible');
        
        notAvailableTimeout = setTimeout(() => {
            featureNameEl.textContent = "";
            featureNameEl.style.color = "";
            featureNameEl.classList.remove('visible');
            notAvailableTimeout = null;
        }, 2000);
    }
    console.log(`No features found with ID: ${targetId}`);
}
}

// Initialize Atlas Viewer
function initializeAtlasViewer() {
getGeoJSON(GEOJSON_URL).then(geojsonData => {
    const geojsonObject = geojsonData;

    vectorLayer = new VectorLayer({
        transition: 0,
        source: new VectorSource({
            format: new GeoJSON(),
            wrapX: false,
        }),
        style: styleFunction,
    });

    const zoomifySource = new Zoomify({
        url: IMAGE_URL,
        size: [72754, 86284], // This should be dynamic based on metadata
        crossOrigin: "anonymous",
        tierSizeCalculation: 'truncated',
        imageSmoothing: false,
        tileSize: 2048
    });

    const imagery = new TileLayer({
        source: zoomifySource
    });

    const extent = zoomifySource.getTileGrid().getExtent();

    map = new Map({
        layers: [imagery, vectorLayer],
        target: 'map',
        view: new View({
            zoom: 10,
            minZoom: 8,
            maxZoom: 19,
            rotation: (90 * Math.PI / 180), // This should be dynamic
            extent: extent
        }),
        controls: ol.control.defaults({
            zoom: false,
            rotate: false,
            attribution: false
        })
    });

    window.map = map; // Make map globally accessible

    map.getView().fit(imagery.getSource().getTileGrid().getExtent());
    var centerMap = map.getView().getCenter();
    vectorSource = vectorLayer.getSource();

    var features = vectorSource.getFormat().readFeatures(geojsonObject);
    features.forEach(element => {
        var elementRotate = element.getGeometry();
        var xy = centerMap;
        elementRotate = elementRotate.rotate(((90) * Math.PI / 180), xy);
    });
    vectorSource.addFeatures(features);
    vectorSource.getFeatures().forEach(element => {
        element.setStyle(styleFunction(element));
    });

    selectInteraction = new Select({
        condition: ol.events.condition.singleClick,
        style: null,
    });

    map.addInteraction(selectInteraction);

    selectInteraction.on('select', async function (e) {
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

            if (notAvailableTimeout) {
                clearTimeout(notAvailableTimeout);
                notAvailableTimeout = null;
            }

            featureNameEl.textContent = ` ${clickedName}`;
            featureNameEl.style.color = "";
            featureNameEl.classList.add('visible');

            const payload = {
                id: session_id,
                tool_name: 'atlas_viewer',
                params: {
                    id: clickedId,
                    name: clickedName,
                    biosample: currentBiosample,
                    section: currentSection
                }
            };
            window.parent.postMessage(payload, '*');

        } else {
            currentSelectedId = null;
            
            if (notAvailableTimeout) {
                clearTimeout(notAvailableTimeout);
                notAvailableTimeout = null;
            }
            
            featureNameEl.textContent = "";
            featureNameEl.style.color = "";
            featureNameEl.classList.remove('visible');

            const payload = {
                id: session_id,
                tool_name: 'atlas_viewer',
                params: {
                    id: -1,
                    name: null,
                    biosample: currentBiosample,
                    section: currentSection
                }
            };
            window.parent.postMessage(payload, '*');
        }
    });

    // Set up opacity slider
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
            vectorLayer.setVisible(true);
            eyeIcon.innerHTML = `
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                <circle cx="12" cy="12" r="3"></circle>
            `;
            eyeIcon.classList.remove('hidden');
            
            if (currentSelectedId !== null) {
                setTimeout(() => {
                    const matchingFeatures = vectorSource.getFeatures().filter(f => {
                        const data = f.get('data');
                        return data && data.id === currentSelectedId;
                    });
                    
                    if (matchingFeatures.length > 0) {
                        selectInteraction.getFeatures().clear();
                        
                        vectorSource.getFeatures().forEach(feature => {
                            feature.setStyle(styleFunction(feature));
                            feature.changed();
                        });
                        
                        selectInteraction.getFeatures().push(matchingFeatures[0]);
                        
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
                            f.changed();
                        });
                        
                        vectorSource.changed();
                        vectorLayer.changed();
                        map.render();
                        
                        const featureData = matchingFeatures[0].get('data');
                        const featureNameEl = document.getElementById("feature-name");
                        if (featureData && featureData.name && featureNameEl) {
                            featureNameEl.textContent = ` ${featureData.name}`;
                            featureNameEl.classList.add('visible');
                        }
                    }
                }, 200);
            }
        } else {
            vectorLayer.setVisible(false);
            eyeIcon.innerHTML = `
                <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
                <line x1="1" y1="1" x2="23" y2="23"></line>
            `;
            eyeIcon.classList.add('hidden');
        }
    });
});
}

// ==================== Thumbnail Viewer Code ====================
let imagesData = [];
let startIndex = 0;
let isAnimating = false;

const gridWrapper = document.querySelector(".image-grid-wrapper");
const grid1 = document.getElementById("image-grid-1");
const grid2 = document.getElementById("image-grid-2");

async function fetchImages() {
try {
    const response = await fetch(THUMBNAIL_API_URL);
    if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
    }
    const data = await response.json();
    const selectedSeries = data.find(item => item.seriesType === currentSeriesType);
    imagesData = selectedSeries?.thumbnails ?? [];
    if (imagesData.length === 0) {
        console.warn("No images found for the selected series.");
        return;
    }
    
    let roiIndex = imagesData.findIndex(img => img.sectionNo == currentSection);

    if (roiIndex === -1) {
        let closestDiff = Infinity;
        let closestIndex = 0;
        for (let i = 0; i < imagesData.length; i++) {
            const diff = Math.abs(imagesData[i].sectionNo - currentSection);
            if (diff < closestDiff) {
                closestDiff = diff;
                closestIndex = i;
            }
        }
        roiIndex = closestIndex;
    }

    const columns = calculateColumns();
    const rows = Math.ceil((window.innerHeight - 100) / 100);
    const imagesPerPage = columns * rows;

    startIndex = roiIndex - Math.floor(imagesPerPage / 2);
    if (startIndex < 0) startIndex = 0;
    if (startIndex > imagesData.length - imagesPerPage) {
        startIndex = Math.max(imagesData.length - imagesPerPage, 0);
    }

    renderImages(grid1, startIndex);
} catch (error) {
    console.error("Error fetching images:", error);
}
}

function calculateColumns() {
const containerWidth = document.querySelector(".image-grid-container").clientWidth;
return Math.floor(containerWidth / 90);
}

function calculateRows() {
const thumbnailPanelHeight = thumbnailPanel.clientHeight - 100; // Account for header
return Math.floor(thumbnailPanelHeight / 90);
}

function createImageElement(image, index) {
const container = document.createElement("div");
container.classList.add("image-container");
container.setAttribute("data-section-no", image.sectionNo);

const img = document.createElement("img");
img.src = image.thumbnailUrl;

img.onerror = function () {
    img.style.display = "none";
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

// Highlight current section
if (image.sectionNo == currentSection) {
    container.classList.add("selected");
}

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

const newSection = parseInt(imageContainer.getAttribute("data-section-no"));

// Update current section and reload atlas viewer
currentSection = newSection;
document.getElementById('atlas-section').textContent = currentSection;

// Clear current atlas data and reload with new section
console.log(`Loading section ${currentSection}`);

// Update the URLs with the new section
const newGeojsonUrl = GEOJSON_URL.replace(/\/\d+\/[^/]+\.json$/, `/${currentSection}/222-NISL-${currentSection}-FlatTree::IIT:V1:SS-100:306:1000:1000.json`);
const newImageUrl = IMAGE_URL.replace(/SE_\d+/, `SE_${currentSection}`);

// Reinitialize atlas viewer with new section data
if (window.map) {
    // Clear existing features
    if (vectorSource) {
        vectorSource.clear();
    }
    
    // Load new GeoJSON data
    getGeoJSON(newGeojsonUrl).then(geojsonData => {
        if (geojsonData && vectorSource) {
            const features = vectorSource.getFormat().readFeatures(geojsonData);
            const centerMap = map.getView().getCenter();
            
            features.forEach(element => {
                const elementRotate = element.getGeometry();
                elementRotate.rotate((90 * Math.PI / 180), centerMap);
            });
            
            vectorSource.addFeatures(features);
            vectorSource.getFeatures().forEach(element => {
                element.setStyle(styleFunction(element));
            });
            
            console.log(`Atlas updated with section ${currentSection}`);
        }
    }).catch(error => {
        console.error('Error updating atlas:', error);
    });
}

const payload = {
    id: session_id,
    tool_name: 'thumbnail_viewer',
    params: {
        biosample: currentBiosample,
        section: currentSection,
    },
};

window.parent.postMessage({ action_context: payload }, "*");
});

function renderImages(grid, index) {
grid.innerHTML = "";

const columns = calculateColumns();
const rowHeight = 100;
const rows = Math.ceil((window.innerHeight - 100) / rowHeight);
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
if (isAnimating || !imagesData.length) return;
isAnimating = true;

const columns = calculateColumns();
const rows = calculateRows();
const imagesPerPage = columns * rows;
let nextIndex = startIndex;

if (direction === "next") {
    nextIndex += imagesPerPage;
} else {
    nextIndex -= imagesPerPage;
}

nextIndex = Math.max(0, Math.min(nextIndex, imagesData.length - imagesPerPage));

if (nextIndex === startIndex) {
    isAnimating = false;
    return;
}

// Use smooth scrolling animation
const container = document.querySelector('.image-grid-container');
const targetGrid = startIndex < nextIndex ? grid2 : grid1;

// Render new images in the target grid
renderImages(targetGrid, nextIndex);

// Create smooth transition
gridWrapper.style.transition = "transform 0.3s ease-in-out";
gridWrapper.style.transform = `translateX(${startIndex < nextIndex ? "-50%" : "50%"})`;

setTimeout(() => {
    gridWrapper.style.transition = "none";
    gridWrapper.style.transform = "translateX(0)";

    // Copy content to main grid
    if (startIndex < nextIndex) {
        grid1.innerHTML = grid2.innerHTML;
    } else {
        grid2.innerHTML = grid1.innerHTML;
    }

    startIndex = nextIndex;
    isAnimating = false;
}, 300);
}

// Navigation buttons removed - using scroll wheel and keyboard only

// Navigation button event listeners removed

window.addEventListener("resize", () => {
const columns = calculateColumns();
const rows = Math.ceil((window.innerHeight - 100) / 100);
const imagesPerPage = columns * rows;

const sortedImages = [...imagesData].sort(
    (a, b) => parseInt(a.sectionNo) - parseInt(b.sectionNo)
);

const maxStart = Math.max(0, sortedImages.length - imagesPerPage);
if (startIndex > maxStart) {
    startIndex = maxStart;
}

renderImages(grid1, startIndex);

if (window.map) {
    window.map.updateSize();
}
});

// Series selector change handler
document.getElementById('series-select').addEventListener('change', function(e) {
currentSeriesType = e.target.value;
fetchImages();
});

// Add scroll wheel support for navigation
document.querySelector('.image-grid-container').addEventListener('wheel', function(e) {
if (isAnimating) return;

const columns = calculateColumns();
const rows = calculateRows();
const imagesPerPage = columns * rows;
const canGoNext = startIndex + imagesPerPage < imagesData.length;
const canGoPrev = startIndex > 0;

if (e.deltaY > 0) {
    // Scroll down - next page
    if (canGoNext) {
        e.preventDefault();
        scrollImages('next');
    }
} else {
    // Scroll up - previous page
    if (canGoPrev) {
        e.preventDefault();
        scrollImages('prev');
    }
}
});

// Add keyboard support
document.addEventListener('keydown', function(e) {
// Only handle keys when focus is on thumbnail area or no input is focused
const focusedElement = document.activeElement;
const isInputFocused = focusedElement.tagName === 'INPUT' || 
                        focusedElement.tagName === 'TEXTAREA' || 
                        focusedElement.contentEditable === 'true';

if (!isInputFocused && !isAnimating) {
    const columns = calculateColumns();
    const rows = calculateRows();
    const imagesPerPage = columns * rows;
    const canGoNext = startIndex + imagesPerPage < imagesData.length;
    const canGoPrev = startIndex > 0;
    
    if (e.key === 'ArrowLeft' && canGoPrev) {
        e.preventDefault();
        scrollImages('prev');
    } else if (e.key === 'ArrowRight' && canGoNext) {
        e.preventDefault();
        scrollImages('next');
    }
}
});

// ==================== Message Handlers ====================
window.addEventListener('message', (event) => {
console.log('Message received:', event.data);

if (event.data.action === 'sayHi') {
    alert('Got this from parent: ' + event.data.data);
}

if (event.data.action === 'highlight' && event.data.region) {
    const { id } = event.data.region;
    console.log(`Triggering click event for region ID: ${id}`);
    triggerClickEventById(id);
}

if (event.data.id && event.data.name && event.data.type == "HighlightRegion" && typeof event.data.id === 'number') {
    console.log(`Triggering click event for region: ${event.data.name} (ID: ${event.data.id})`);
    triggerClickEventById(event.data.id);
}

// Handle updates from parent
if (event.data.action === 'updateViewer' && event.data.params) {
    const { biosample, section, seriesType } = event.data.params;
    if (biosample) {
        currentBiosample = biosample;
        document.getElementById('thumbnail-biosample').textContent = biosample;
        document.getElementById('atlas-biosample').textContent = biosample;
    }
    if (section) {
        currentSection = section;
        document.getElementById('atlas-section').textContent = section;
    }
    if (seriesType) {
        currentSeriesType = seriesType;
        document.getElementById('series-select').value = seriesType;
    }
    // Reinitialize viewers with new data
    fetchImages();
    initializeAtlasViewer();
}
});

// ==================== Initialize ====================
// Initialize both viewers when page loads
window.addEventListener('load', function() {
initializeAtlasViewer();
fetchImages();

// Initialize theme
initializeTheme();

// Clear chat history on page refresh
clearChatHistoryOnRefresh();
});

// ==================== Chat History Clear on Refresh ====================
function clearChatHistoryOnRefresh() {
// Clear chat history on page refresh
try {
    fetch('/api/chat-history', {
        method: 'DELETE'
    }).then(response => {
        if (response.ok) {
            console.log('Chat history cleared on page refresh');
        }
    }).catch(error => {
        console.error('Error clearing chat history on refresh:', error);
    });
} catch (error) {
    console.error('Error clearing chat history on refresh:', error);
}

// Also clear local UI state
if (chatMessages) {
    chatMessages.innerHTML = '';
}
currentBrainRegion = null;
if (currentRegionSpan) {
    currentRegionSpan.textContent = '';
}
if (askBtn) {
    askBtn.disabled = true;
}
}

// ==================== Theme Toggle Functionality ====================
function initializeTheme() {
// Check for saved theme preference or default to dark theme
const savedTheme = localStorage.getItem('theme') || 'dark';

if (savedTheme === 'light') {
    document.body.classList.add('light-theme');
}
}

// Theme toggle button click handler
document.getElementById('themeToggle').addEventListener('click', function() {
const isLight = document.body.classList.contains('light-theme');

if (isLight) {
    // Switch to dark theme
    document.body.classList.remove('light-theme');
    localStorage.setItem('theme', 'dark');
} else {
    // Switch to light theme
    document.body.classList.add('light-theme');
    localStorage.setItem('theme', 'light');
}

// Optional: Refresh map to apply theme changes properly
if (window.map) {
    setTimeout(() => {
        window.map.updateSize();
    }, 300);
}
});

// ==================== Chatbot Functionality ====================
let currentBrainRegion = null;
let isWaitingForResponse = false;

// DOM elements for chatbot
// chatbotColumn already declared in resize section above
const chatbotContainer = document.getElementById('chatbotContainer');
const chatbotToggle = document.getElementById('chatbotToggle');
const chatbotClose = document.getElementById('chatbotClose');
const chatMessages = document.getElementById('chatMessages');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const currentRegionSpan = document.getElementById('currentRegion');
const historyBtn = document.getElementById('historyBtn');
const clearBtn = document.getElementById('clearBtn');
const historyModal = document.getElementById('historyModal');
const closeHistoryModal = document.getElementById('closeHistoryModal');
const closeHistoryBtn = document.getElementById('closeHistoryBtn');
const historyList = document.getElementById('historyList');
const historyStats = document.getElementById('historyStats');
const downloadHistory = document.getElementById('downloadHistory');

// Initialize chatbot event listeners
chatbotToggle.addEventListener('click', () => {
const isActive = chatbotColumn.classList.contains('active');
if (isActive) {
    closeChatbot();
} else {
    openChatbot();
}
});

chatbotClose.addEventListener('click', () => {
closeChatbot();
});

// Handle escape key to close chatbot
document.addEventListener('keydown', (e) => {
if (e.key === 'Escape' && chatbotColumn.classList.contains('active')) {
    closeChatbot();
}
});

function openChatbot() {
mainContainer.classList.add('chatbot-active');
chatbotColumn.classList.add('active');
chatbotToggle.classList.add('hidden');

// Trigger layout updates after animation completes
setTimeout(() => {
    if (window.map) {
        window.map.updateSize();
    }
    // Force reflow and recalculate thumbnail grid
    if (imagesData.length > 0) {
        // Force browser to recalculate layout
        document.querySelector('.thumbnail-panel').offsetHeight;
        renderImages(grid1, startIndex);
        updatePaginationButtons();
    }
}, 450);

// Focus on input if there's a selected region
if (currentBrainRegion && questionInput) {
    setTimeout(() => questionInput.focus(), 400);
}
}

function closeChatbot() {
mainContainer.classList.remove('chatbot-active');
chatbotColumn.classList.remove('active');
chatbotToggle.classList.remove('hidden');

// Trigger layout updates after animation completes
setTimeout(() => {
    if (window.map) {
        window.map.updateSize();
    }
    // Force reflow and recalculate thumbnail grid
    if (imagesData.length > 0) {
        // Force browser to recalculate layout
        document.querySelector('.thumbnail-panel').offsetHeight;
        renderImages(grid1, startIndex);
        updatePaginationButtons();
    }
}, 450);
}


askBtn.addEventListener('click', handleAskQuestion);
questionInput.addEventListener('keypress', (e) => {
if (e.key === 'Enter' && !askBtn.disabled) {
    handleAskQuestion();
}
});

// History button
historyBtn.addEventListener('click', () => {
showChatHistory();
});

// Clear history button
clearBtn.addEventListener('click', () => {
clearChatHistory();
});

// History modal close handlers
closeHistoryModal.addEventListener('click', () => {
historyModal.style.display = 'none';
});

closeHistoryBtn.addEventListener('click', () => {
historyModal.style.display = 'none';
});

// Download history button
downloadHistory.addEventListener('click', () => {
downloadChatHistory();
});

// Close modal when clicking outside
historyModal.addEventListener('click', (e) => {
if (e.target === historyModal) {
    historyModal.style.display = 'none';
}
});

// Enable/disable ask button based on input
questionInput.addEventListener('input', () => {
if (currentBrainRegion && questionInput.value.trim()) {
    askBtn.disabled = false;
} else {
    askBtn.disabled = true;
}
});


async function selectBrainRegion(regionName) {
if (isWaitingForResponse) return;

// Show chatbot
openChatbot();

// Update UI
currentBrainRegion = regionName;
currentRegionSpan.textContent = ` - ${regionName}`;
askBtn.disabled = false;

// Add user message
addMessage(`Tell me about the ${regionName}`, 'user');

// Show loading
const loadingMsg = addMessage('', 'bot');
loadingMsg.innerHTML = '<div class="loading"></div> Analyzing brain region...';

// Get selected mode
const mode = document.querySelector('input[name="mode"]:checked').value;

isWaitingForResponse = true;

try {
    // Use streaming endpoint
    const response = await fetch('/api/brain-region-stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                region: regionName,
                mode: mode
            })
        });
        
        // Remove loading message
        loadingMsg.remove();
        
        // Create a new message for streaming content
        const streamMsg = addMessage('', 'bot');
        let fullMessage = '';
        
        // Read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.chunk) {
                            fullMessage += data.chunk;
                            streamMsg.innerHTML = formatMessage(fullMessage);
                            // Scroll to bottom
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        }
                        if (data.complete && !data.success) {
                            currentBrainRegion = null;
                            currentRegionSpan.textContent = '';
                            askBtn.disabled = true;
                        }
                    } catch (e) {
                        console.error('Error parsing stream data:', e);
                    }
                }
            }
        }
} catch (error) {
    loadingMsg.remove();
    addMessage(`Error: ${error.message}`, 'bot');
} finally {
    isWaitingForResponse = false;
}
}

async function handleAskQuestion() {
const question = questionInput.value.trim();
if (!question || !currentBrainRegion || isWaitingForResponse) return;

// Add user message
addMessage(question, 'user');

// Clear input
questionInput.value = '';

// Show loading
const loadingMsg = addMessage('', 'bot');
loadingMsg.innerHTML = '<div class="loading"></div> Thinking...';

// Get selected mode to determine if web search should be used
const mode = document.querySelector('input[name="mode"]:checked').value;
const useWeb = (mode === 'web'); // Automatically use web for 'Detailed (Web)' mode

isWaitingForResponse = true;

try {
    // Use streaming endpoint
    const response = await fetch('/api/ask-question-stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                use_web: useWeb,
                mode: mode
            })
        });
        
        // Remove loading message
        loadingMsg.remove();
        
        // Create a new message for streaming content
        const streamMsg = addMessage('', 'bot');
        let fullMessage = '';
        
        // Read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.chunk) {
                            fullMessage += data.chunk;
                            streamMsg.innerHTML = formatMessage(fullMessage);
                            // Scroll to bottom
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        }
                        if (data.error) {
                            streamMsg.innerHTML = formatMessage(`Error: ${data.error}`);
                        }
                    } catch (e) {
                        console.error('Error parsing stream data:', e);
                    }
                }
            }
        }
    
} catch (error) {
    loadingMsg.remove();
    addMessage(`Error: ${error.message}`, 'bot');
} finally {
    isWaitingForResponse = false;
}
}

function formatMessage(text) {
// Convert markdown formatting to HTML
let formattedText = text
    // Handle markdown headers (###, ##, #)
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Handle numbered lists
    .replace(/^(\d+)\. (.+)$/gm, '<br>$1. $2')
    // PRIORITY 0: Exact problematic patterns first
    .replace(/^(Khan Academy Academic Search|Educational & Academic Sources|Medical & Clinical Databases|Alternative Search Results):/gim, '<strong>$1:</strong>')
    // PRIORITY 1: Specific source names (catch them before other patterns)
    .replace(/^(Wikipedia|PubMed|Google|Bing|Yahoo|DuckDuckGo|Yandex|Khan Academy|Educational|Medical|Clinical|Academic|Scientific|Alternative|Extended|Research|University|Nature|Science|NCBI|NIH|FDA|WHO)(\s+[A-Za-z\s&]*)?(?:\s*-\s*[^:]+)?:/gim, '<strong>$1$2:</strong>')
    // PRIORITY 2: Source - Topic format (like "Wikipedia - Cerebral cortex:")
    .replace(/^([A-Z][A-Za-z\s&]+)\s*-\s*([^:]+):/gm, '<strong>$1 - $2:</strong>')
    // PRIORITY 3: Any line with search/academic keywords
    .replace(/^([A-Z][^:\n]*?(?:Search|Results?|API|Database|Sources?|Engine|Web|Site|Portal|Research|Studies|Papers|Articles|Information|Data|Repository|Archive|Library|Index|Registry|Collection|Wikipedia|PubMed|Google|Bing|Yahoo|DuckDuckGo|Yandex|Searx|Startpage|Ecosia|Nature|Science|NCBI|NIH|FDA|WHO|Medical|Clinical|Journal|University|Academic|Study|Report|Article|Paper|Publication|Summary|Overview|Definition|Findings|Analysis|Review|Investigation|Khan|Academy|Educational|Extended|Alternative)[^:\n]*?(?:\s*\([^)]*\))?)\s*:/gim, '<strong>$1:</strong>')
    // PRIORITY 4: Multi-word headings with & (like "Educational & Academic Sources:")
    .replace(/^([A-Z][A-Za-z\s&-]{8,})\s*:/gm, '<strong>$1:</strong>')
    // PRIORITY 5: Catch ANY capitalized heading with parentheses
    .replace(/^([A-Z][A-Za-z\s&-]{3,}(?:\s*\([^)]*\)))\s*:/gim, '<strong>$1:</strong>')
    // PRIORITY 6: Simple capitalized headings (3+ words)
    .replace(/^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){2,}(?:\s*\([^)]*\))?)\s*:/gim, '<strong>$1:</strong>')
    // PRIORITY 7: Any line that starts with capital and ends with colon
    .replace(/^([A-Z][A-Za-z\s&-]{5,})\s*:\s*(?=[A-Z•])/gm, '<strong>$1:</strong>')
    // PRIORITY 8: Common research terms
    .replace(/^(Sources?|References?|From|Based on|According to|Research shows|Studies indicate|Key findings|Summary|Overview|Definition|Clinical significance|Anatomy|Function|Location|Connections|Role in disease|Results|Findings|Conclusion|Abstract|Introduction|Methods|Discussion|Background|Objective|Purpose|Aim|Goal|Analysis|Review|Investigation|Extended|Alternative|Academic|Scientific|Educational|Additional|Recent|Current|Latest|Updated)(?:\s+[A-Za-z\s&]*)?:/gim, '<strong>$1:</strong>')
    // PRIORITY 9: Last resort - any reasonable heading format
    .replace(/^([A-Z][A-Za-z\s&]{2,})\s*:/gm, '<strong>$1:</strong>')
    // Handle bullet points: * item -> • item
    .replace(/^\* (.*$)/gm, '• $1')
    // Handle bullet points: + item -> • item
    .replace(/^\+ (.*$)/gm, '• $1')
    // Handle bullet points: - item -> • item
    .replace(/^- (.*$)/gm, '• $1')
    // Bold text: **text** -> <strong>text</strong>
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // Underline: __text__ -> <u>text</u>
    .replace(/__(.*?)__/g, '<u>$1</u>')
    // Convert line breaks to <br>
    .replace(/\n/g, '<br>');

return formattedText;
}

function addMessage(text, type) {
const messageDiv = document.createElement('div');
messageDiv.className = `message ${type}-message`;

const messageP = document.createElement('p');
messageP.innerHTML = formatMessage(text);

messageDiv.appendChild(messageP);
chatMessages.appendChild(messageDiv);

// Scroll to bottom
chatMessages.scrollTop = chatMessages.scrollHeight;

return messageP;
}

async function showChatHistory() {
try {
    const response = await fetch('/api/chat-history');
    const data = await response.json();
    
    if (data.success) {
        displayChatHistory(data.chat_history, data.current_region, data.current_mode);
        historyModal.style.display = 'block';
    } else {
        alert('Error loading chat history: ' + data.message);
    }
} catch (error) {
    alert('Error loading chat history: ' + error.message);
}
}

function displayChatHistory(history, currentRegion, currentMode) {
// Update stats
const totalMessages = history.length;
const userQuestions = history.filter(msg => msg.type === 'user_query').length;
const regions = [...new Set(history.filter(msg => msg.region).map(msg => msg.region))];

historyStats.innerHTML = `
    <strong>Chat Statistics:</strong> 
    ${totalMessages} total messages, 
    ${userQuestions} questions asked, 
    ${regions.length} regions explored
    ${currentRegion ? `<br><strong>Current:</strong> ${currentRegion} (${currentMode || 'fast'} mode)` : ''}
`;

// Clear previous history
historyList.innerHTML = '';

if (history.length === 0) {
    historyList.innerHTML = '<div class="no-history">No chat history yet. Start exploring brain regions!</div>';
    return;
}

// Group messages by region for better organization
const groupedHistory = {};
history.forEach((msg, index) => {
    const region = msg.region || 'General';
    if (!groupedHistory[region]) {
        groupedHistory[region] = [];
    }
    groupedHistory[region].push({...msg, index});
});

// Display grouped history
Object.keys(groupedHistory).forEach(region => {
    const regionDiv = document.createElement('div');
    regionDiv.className = 'history-region-group';
    
    const regionHeader = document.createElement('h4');
    regionHeader.className = 'history-region-header';
    regionHeader.textContent = region;
    regionDiv.appendChild(regionHeader);
    
    groupedHistory[region].forEach(msg => {
        const historyItem = document.createElement('div');
        historyItem.className = `history-item ${msg.type}`;
        
        const timestamp = new Date(msg.timestamp).toLocaleString();
        const typeLabel = {
            'user_query': '❓ Question',
            'region_info': '🧠 Region Info', 
            'assistant_response': '🤖 Answer'
        }[msg.type] || '💬 Message';
        
        historyItem.innerHTML = `
            <div class="history-item-header">
                <span class="history-type">${typeLabel}</span>
                <span class="history-timestamp">${timestamp}</span>
            </div>
            <div class="history-content">${msg.content}</div>
        `;
        
        regionDiv.appendChild(historyItem);
    });
    
    historyList.appendChild(regionDiv);
});
}

async function downloadChatHistory() {
try {
    const response = await fetch('/api/chat-history');
    const data = await response.json();
    
    if (data.success) {
        const historyText = formatHistoryForDownload(data.chat_history);
        const blob = new Blob([historyText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `brain-assistant-history-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } else {
        alert('Error downloading chat history: ' + data.message);
    }
} catch (error) {
    alert('Error downloading chat history: ' + error.message);
}
}

function formatHistoryForDownload(history) {
let text = 'Brain Assistant Chat History\n';
text += '================================\n';
text += `Generated on: ${new Date().toLocaleString()}\n\n`;

let currentRegion = '';

history.forEach((msg, index) => {
    const timestamp = new Date(msg.timestamp).toLocaleString();
    
    // Add region header when it changes
    if (msg.region && msg.region !== currentRegion) {
        currentRegion = msg.region;
        text += `\n--- ${currentRegion.toUpperCase()} ---\n`;
    }
    
    const typeLabel = {
        'user_query': 'USER',
        'region_info': 'ASSISTANT (Region Info)',
        'assistant_response': 'ASSISTANT'
    }[msg.type] || 'MESSAGE';
    
    text += `[${timestamp}] ${typeLabel}:\n${msg.content}\n\n`;
});

return text;
}

async function clearChatHistory() {
if (!confirm('Are you sure you want to clear all chat history? This cannot be undone.')) {
    return;
}

try {
    const response = await fetch('/api/chat-history', {
        method: 'DELETE'
    });
    
    const data = await response.json();
    
    if (data.success) {
        // Clear the chat messages UI
        chatMessages.innerHTML = '';
        currentBrainRegion = null;
        currentRegionSpan.textContent = '';
        askBtn.disabled = true;
        alert('Chat history cleared successfully!');
        // Close history modal if open
        historyModal.style.display = 'none';
    } else {
        alert('Error clearing chat history: ' + data.message);
    }
} catch (error) {
    alert('Error clearing chat history: ' + error.message);
}
}

// Integration with atlas viewer - when a region is clicked
window.addEventListener('message', (event) => {
if (event.data.tool_name === 'atlas_viewer' && event.data.params) {
    const { name, id } = event.data.params;
    if (name && id !== -1) {
        // Auto-select the brain region in chatbot
        selectBrainRegion(name);
    }
}
});

