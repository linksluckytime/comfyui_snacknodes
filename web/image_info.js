import { app } from "../../scripts/app.js";

// 为ImageInfo节点添加显示值的功能
app.registerExtension({
    name: "comfyui.snacknodes.imageinfo",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 只处理ImageInfo节点
        if (nodeData.name !== "ImageInfo") {
            return;
        }

        // 保存原始的onExecuted方法
        const onExecuted = nodeType.prototype.onExecuted;

        // 重写onExecuted方法
        nodeType.prototype.onExecuted = function(message) {
            // 调用原始方法
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }

            // 从消息中获取输出值
            if (message && message.output) {
                const outputs = message.output;
                
                // 为每个输出创建或更新标签
                if (outputs.length >= 4) {
                    const width = outputs[0];
                    const height = outputs[1];
                    const batch_size = outputs[2];
                    const channels = outputs[3];

                    // 显示计算出的值
                    this.title = `Image Info 🍿
宽度: ${width}
高度: ${height}
批次: ${batch_size}
通道: ${channels}`;
                    
                    // 告诉节点更新显示
                    this.setDirtyCanvas(true, true);
                }
            }
        };
    }
}); 