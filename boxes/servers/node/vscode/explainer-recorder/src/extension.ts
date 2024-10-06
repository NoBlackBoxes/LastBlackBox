import * as vscode from 'vscode';
import Microphone from 'node-microphone';
import * as fs from 'fs';
import * as path from 'path';

export function activate(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.commands.registerCommand('explainer-recorder', () => {
            const resources_folder_uri: vscode.Uri = vscode.Uri.joinPath(context.extensionUri, 'resources');

            const panel = vscode.window.createWebviewPanel(
                'explainer-recorder',
                'Explainer Recorder',
                vscode.ViewColumn.Two,
				{
					enableScripts: true,
					localResourceRoots: [resources_folder_uri]
				}
            );

			// Get resources folder *webview* URI
			const resources_folder_webview_uri = panel.webview.asWebviewUri(resources_folder_uri);

			// Set webview HTML content
            panel.webview.html = getWebviewContent(resources_folder_webview_uri);

			// Handle messages from the webview
            panel.webview.onDidReceiveMessage(message => {
                switch (message.command) {
                    case 'startRecording':
                        startMicrophoneRecording(panel.webview);
                        break;
                    case 'stopRecording':
                        stopMicrophoneRecording(panel.webview);
                        break;
                }
            });
        })
    );
}

let microphone: any;
let micStream: any;
let outputFileStream: fs.WriteStream;
let outputPath: string;

// Define a custom type that extends the existing MicrophoneOptions
interface CustomMicrophoneOptions {
    useDataEmitter?: boolean;
    channels?: 2 | 1 | undefined;
    rate?: 8000 | 16000 | 44100 | undefined;
    bitwidth?: 8 | 16 | 24 | undefined;
    encoding?: "signed-integer" | "unsigned-integer" | undefined;
    endian?: 'big' | 'little';
    additionalParameters?: string[];
}

/**
 * Starts recording from the microphone and saves the audio to a file.
 */
function startMicrophoneRecording(webview: vscode.Webview) {
    outputPath = path.join(__dirname, 'output.wav');

    // Create a new Microphone instance with extended options
    microphone = new Microphone({
        channels: 2,        // Stereo
        rate: 44100,        // Sample rate of 44100 Hz
        bitwidth: 16,       // 16-bit audio
        encoding: 'signed-integer', // Signed integer encoding
        endian: 'little',   // Correctly typed endian value
        useDataEmitter: true,  // Enable the data emitter
        additionalParameters: ['-D', 'plughw:0'],  // Custom device string passed to arecord
    } as CustomMicrophoneOptions);  // Cast to the extended interface

    outputFileStream = fs.createWriteStream(outputPath);  // Create a write stream for output file

    micStream = microphone.startRecording();  // Start recording

    // Handle audio data events
    micStream.on('data', (chunk: Buffer) => {
        console.log("Received audio chunk:", chunk);
    });

    // Pipe the audio stream to the file
    micStream.pipe(outputFileStream);

    console.log('Recording started');

    //microphone.on('info', (info) => {
    //    console.log(info);
    //});
    //
    //microphone.on('error', (error) => {
    //    console.log(error);
    //});
}

/**
 * Stops the microphone recording and sends the audio file path back to the webview.
 */
function stopMicrophoneRecording(webview: vscode.Webview) {
    if (micStream && microphone) {
        // Stop piping the stream to the file
        micStream.unpipe(outputFileStream);
        
        // Stop the recording using the microphone instance
        microphone.stopRecording();  // This is the correct method to stop recording

        // Ensure the file is finalized and closed properly
        outputFileStream.end(() => {
            console.log("Recording stopped");

            // Send the recorded file back to the webview for playback
            const fileUri = vscode.Uri.file(outputPath);
            webview.postMessage({ command: 'audioRecorded', fileUri: webview.asWebviewUri(fileUri) });
        });
    }
}

/**
 * Provides the HTML content for the webview, including buttons for recording and stopping.
 */
function getWebviewContent(resources_uri: vscode.Uri): string {
    return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Explainer Recorder</title>
            <script src="${vscode.Uri.joinPath(resources_uri, 'explainer_recorder.js')}"></script>
        </head>
        <body>
            <hr>
            <button id="start-button">Start Capture</button>
            <button id="stop-button">Stop Capture</button>
            <hr>
        </body>
        </html>
    `;
}
