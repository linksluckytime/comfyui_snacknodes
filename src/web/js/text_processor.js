import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// 注册扩展以显示文本处理结果
app.registerExtension({
    name: "comfyui_snacknodes.TextProcessor",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "TextProcessor") {
            const onExecuted = nodeType.prototype.onExecuted;
            
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                // 清理现有的结果widget
                if (this.widgets) {
                    const pos = this.widgets.findIndex((w) => w.name === "result");
                    if (pos !== -1) {
                        for (let i = pos; i < this.widgets.length; i++) {
                            this.widgets[i].onRemove?.();
                        }
                        this.widgets.length = pos;
                    }
                }
                
                // 创建新的结果widget
                const w = ComfyWidgets["STRING"](this, "result", ["STRING", { multiline: true }], app).widget;
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.value = message.text;
                
                // 调整节点大小
                this.onResize?.(this.size);
            };
        }
    },
}); 