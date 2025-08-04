import openai
from openai import OpenAI
from langchain.sql_database import SQLDatabase
import ast
# from utils.agent.main import main_tool_cache
from ..urls import base_url
from ..shared_config import get_brain_and_name_context
# from .context
from sqlalchemy import create_engine, text

base_url = "http://dgx3.humanbrain.in:10603"

# MySQL
MySQL_db_user = "root"
MySQL_db_password = "Health#123"
db_host = "dev2mani.humanbrain.in"
# db_host = "dev2kamal.humanbrain.in"
MySQL_db_port = "3306"
MySQL_db_name = "HBA_V2"
# db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

MySQL_DATABASE_URL = f"mysql+pymysql://{MySQL_db_user}:{MySQL_db_password}@{MySQL_db_host}:{MySQL_db_port}/{MySQL_db_name}"
MySQL_engine = create_engine(MySQL_DATABASE_URL)

def MySQL_db_retriever(sql_query):
    with MySQL_engine.connect() as connection:  
        result = connection.execute(text(sql_query))  
    data = result.fetchall()
    return data

thumbnail_cache = {}

mappings_table = get_brain_and_name_context()

def get_ssid(biosample_id):
    query = f"SELECT * FROM HBA_V2.seriesset where biosample={biosample_id};"
    result = MySQL_db_retriever(query)
    return result[0][0]

def get_stain_map(mnemonic_name):
    result = MySQL_db_retriever(f"""
    SELECT name FROM HBA_V2.seriestype where mnemonic='{mnemonic_name}';
    """)
    return result[0][0]


def llm_resp(user_query, chat_history=None, page_context=None):
    client = OpenAI(base_url="http://dgx5.humanbrain.in:8999/v1", api_key="dummy")
    json_structure = {"biosample":"","series":""}
    print("PAGE CONTEXT INSDE GET THUMBNAILS", page_context)
    # print(mappings_table)
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
    system: {'biosample':222,'series':'NISSL', 'roi_section':916}

    User: open thumbnails of brain 244
    page context:{'ssid': 100, 'seriesType': 'NISSL', 'secid': 916, 'biosampleId': '244'}
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
    # print(prompt_template)
    
    messages = [
        {"role": "system", "content": prompt_template}]

    completion = client.chat.completions.create(
        model="Llama-3.3-70B-Instruct",
        messages=messages,
        # tools=tools,
        temperature=0,
    )

    return completion.choices[0].message.content



def thumbnail_viewer(query, session_id, chat_history=None, page_context=None):
    print("thumbnail viewer")
    
    print("query:", query)
    print(page_context)
    llm_reponse = llm_resp(query, chat_history, page_context)
    print("THUMBNAIL VIEWER LLM RESPONSE:", llm_reponse)
    response = ast.literal_eval(llm_reponse)

    print("THUMBNAIL VIEWER RESPONSE:", response)
    biosample, series_type, roi_section = response['biosample'], response['series'], response['roi_section']

    return f"{base_url}/thumbnail/viewer/{biosample}/{series_type}/{session_id}/{roi_section}"

# res = thumbnail_viewer("open 222 thummbnails")
# print(res)


def get_thumbnail_viewer(biosample, series_type, session_id, roi_section=1):
    # return "http://dgx3.humanbrain.in:10607/thumbnail/viewer"
    # if roi_section==None:
    #     roi_section = 1
    print("biosample: ", biosample, "series_type: ", series_type, "session_id: ", session_id, "roi_section: ", roi_section)
    # return "http://dgx3.humanbrain.in:10605/thumbnail/viewer"
    ss_id = get_ssid(biosample)
    stain_name = get_stain_map(series_type)
    print(ss_id)
    # if biosample in thumbnail_cache.keys():
    #     if series_type in thumbnail_cache[biosample].keys():
    #         print("Rendering thumbnail  viewer from cache")
    #         template = thumbnail_cache[biosample][series_type] 
    # else:
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
    with open("/apps/src/utils/agent/tools/actions/get_thumbnails/thumbnail_viewer.html", "w") as file:
        file.write(template)

    # print("saved")
    return template
    # if biosample not in main_tool_cache['thumbnail_viewer'].keys():
    #     main_tool_cache['thumbnail_viewer'][biosample] = {}
    #     if series_type not in main_tool_cache['thumbnail_viewer'][biosample].keys():
    #         main_tool_cache['thumbnail_viewer'][biosample][series_type] = template

    # return "http://dgx3.humanbrain.in:10607/thumbnail/viewer"
    # return "http://dgx3.humanbrain.in:10605/thumbnail/viewer/"

    # return f"http://dgx3.humanbrain.in:10605/thumbnail/viewer/{biosample}/{series_type}"
    # return "http://dgx3.humanbrain.in:10606/thumbnail/atlas/viewer"
