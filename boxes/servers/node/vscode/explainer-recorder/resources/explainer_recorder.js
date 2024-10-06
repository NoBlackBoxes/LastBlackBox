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

    // Report details
    document.getElementById('report-button').addEventListener('click', () => {
        vscode.postMessage({ command: 'reportDetails' });
        console.log("reporting");  // Log when the report button is clicked
    });

    // Handle message from the extension
    window.addEventListener('message', event => {
        const message = event.data;
        if (message.command === 'audioRecorded') {
            console.log("Audio file saved:", message.fileUri);  // Log the file URI
            // You can handle any post-save action here if needed
        }
        else if (message.command === 'updateDetails') {
            details = document.getElementById('details-text').innerText = message.status;
        }
    });
};
