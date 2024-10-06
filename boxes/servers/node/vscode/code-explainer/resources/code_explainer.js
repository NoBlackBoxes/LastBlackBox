window.onload = () => {
    const vscode = acquireVsCodeApi();
    const audio = document.getElementById('audioPlayer');

    // Check if the audio element was properly found
    if (audio) {
        // Send the current playback position to the extension every second
        setInterval(() => {
            const currentTime = audio.currentTime;
            vscode.postMessage({ command: 'updatePosition', currentTime });
        }, 1000);

        // Notify when the audio ends
        audio.onended = () => {
            vscode.postMessage({ command: 'audioEnded' });
        };
    } else {
        console.error("Audio element not found!");
    }
};
