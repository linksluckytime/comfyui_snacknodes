app.registerExtension({
	name: "example.imageselector",
    async setup() {
        function messageHandler(event) { alert(event.detail.message); }
        app.api.addEventListener("example.imageselector.textmessage", messageHandler);
    },
})