import {Streamlit, RenderData} from "streamlit-component-lib"
import G6 from "@antv/g6"

const graphContainer = document.body.appendChild(document.createElement("div"))
graphContainer.style.width = "100%";
graphContainer.style.minHeight = "600px";

function featuresToHTML(features: any): string {
    let html = "";
    if (Object.keys(features).length > 0) {
        html += "<h8>Features</h8><ul>"
        for (const k in features) {
            html += `<li>${k}: ${features[k]}</li>`
        }
        html += "</ul>"
    }
    return html
}

async function onRender(event: Event): Promise<void> {
    const data = (event as CustomEvent<RenderData>).detail

    const graphJson = data.args["graphJson"]

    const nodeTooltip = new G6.Tooltip({
        offsetX: 10,
        offsetY: 10,
        getContent(e: any) {
            const div = document.createElement("div");
            // div.style.width = "100px";
            const model = e.item.getModel();
            div.innerHTML = `
                <h7>Node-Info</h7>
                <ul>
                    <li>Id: ${model.id}</li>
                    <li>Type: ${model.nodeType}</li>
                </ul>
            `
            div.innerHTML += featuresToHTML(model.features);
            return div
        },
        itemTypes: ["node"]
    })
    const edgeTooltip = new G6.Tooltip({
        offsetX: 10,
        offsetY: 10,
        getContent(e: any) {
            const div = document.createElement("div");
            const model = e.item.getModel()
            let sourceNode, targetNode
            for (let i = 0; i < graphJson["nodes"].length; i++) {
                const node = graphJson["nodes"][i]
                if (node["id"] === model.source) {
                    sourceNode = node;
                } else if (node["id"] === model.target) {
                    targetNode = node;
                }
            }
            div.innerHTML = `
                <h7>Edge-Info</h7>
                <ul>
                    <li>From: ${sourceNode["label"]} (${model.source}), 
                        To: ${targetNode["label"]} (${model.target})</li>
                    <li>Type: ${model.edgeType || "no type"}</li>
                    <li>Weight: ${model.weight}</li>
                </ul>
            `
            div.innerHTML += featuresToHTML(model.features);
            return div
        },
        itemTypes: ["edge"]
    })

    const toolbar = new G6.ToolBar();
    const edgeBundling = new G6.Bundling();

    // noinspection TypeScriptValidateTypes
    const graph = new G6.Graph({
        container: graphContainer,
        width: graphContainer.offsetWidth,
        height: graphContainer.offsetHeight,
        plugins: [
            nodeTooltip,
            edgeTooltip,
            toolbar,
            edgeBundling
        ],
        modes: {
            default: [
                "drag-canvas",
                "zoom-canvas",
                "drag-node"
            ]
        },
        defaultNode: {
            style: {
                stroke: "black",
                lineWidth: 1,
                fill: "blue"
            },
            labelCfg: {
                style: {
                    fill: "white"
                },
                autoRotate: true,
            }
        },
        defaultEdge: {
            type: "quadratic",
            style: {
                lineWidth: 0.4,
                opacity: 1,
                endArrow: {
                    path: G6.Arrow.triangle(2.5, 2.5, 0)
                },
                stroke: "black"
            },
            labelCfg: {
                autoRotate: true
            },
        },
    })

    graph.data(graphJson)

    graph.render()
    Streamlit.setFrameHeight()
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
Streamlit.setComponentReady()
Streamlit.setFrameHeight()
