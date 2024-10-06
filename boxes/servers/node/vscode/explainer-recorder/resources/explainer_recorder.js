window.onload = () => {
    const vscode = acquireVsCodeApi();

    // Start recording
    document.getElementById('start-button').addEventListener('click', () => {
        vscode.postMessage({ command: 'startRecording' });
        console.log("started");  // Log when the start button is clicked
    });

    // Stop recording
    document.getElementById('stop-button').addEventListener('click', () => {
        vscode.postMessage({ command: 'stopRecording' });
        console.log("stopped");  // Log when the stop button is clicked
    });

    // Handle message from the extension
    window.addEventListener('message', event => {
        const message = event.data;
        if (message.command === 'audioRecorded') {
            console.log("Audio file saved:", message.fileUri);  // Log the file URI
            // You can handle any post-save action here if needed
        }
    });
};
