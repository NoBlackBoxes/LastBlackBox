// Imports
import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import Microphone from 'node-microphone';

// Activate Extension
export function activate(context: vscode.ExtensionContext) {

    const provider = new ExplainerRecorderProvider(context.extensionUri);

    // Register View
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(ExplainerRecorderProvider.viewType, provider)
    );
}

// ExplainerRecorderProvider Class
class ExplainerRecorderProvider implements vscode.WebviewViewProvider {

    public static readonly viewType = 'explainer-recorder';

    private _view?: vscode.WebviewView;

    constructor(
        private readonly _extensionUri: vscode.Uri,
    ) { }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {

        this._view = webviewView;

        webviewView.webview.options = {
            // Allow scripts in the webview
            enableScripts: true,

            localResourceRoots: [
                this._extensionUri
            ]
        };

        // Get HTML for webview controls
        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Get the active text editor
        const editor = vscode.window.activeTextEditor;
        var outputFilePath: fs.PathLike;
        var can_record: boolean;
        var recording: boolean;
        if (editor) {
            // Get the active file
            var currentFilePath = editor.document.fileName;
            var currentFileName = path.basename(currentFilePath, path.extname(currentFilePath));

            // Set the output file
            outputFilePath = path.join(path.dirname(currentFilePath), currentFileName + '.wav')
            can_record = true;
            recording = false;
        }
        else {
            can_record = false;
            recording = false;
        }

        // Listen for text selection changes
        vscode.window.onDidChangeTextEditorSelection((event) => {
            if (editor) {

                // Get selected text
                const selection = editor.document.getText(editor.selection);
                
                // If there is a selection, send it to the webview
                if (selection) {
                    webviewView.webview.postMessage({ 
                        command: 'updateDetails', 
                        status: `Selected Text: ${selection}`
                    });
                }
            }
        });

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage(message => {
            switch (message.command) {
                case 'startRecording':
                    if (can_record) {
                        startMicrophoneRecording(webviewView.webview, outputFilePath);
                        recording = true;
                        can_record = false;
                    }
                    break;
                case 'stopRecording':
                    if (recording) {
                        stopMicrophoneRecording(webviewView.webview);
                        recording = false;
                        can_record = true;
                    }
                    break;
                case 'reportDetails':
                    this.reportDetails(webviewView.webview);
                    break;
            }
        });
    }

    public reportDetails(webview: vscode.Webview) {

        // Get the active text editor
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // Get the active file
            var currentFilePath = editor.document.fileName;
            var currentFileName = path.basename(currentFilePath);
            console.log(currentFileName)
            webview.postMessage({ command: 'updateDetails', status: "Stuff!" });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {

        // Get resources folder URI
        const resources_folder_uri: vscode.Uri = vscode.Uri.joinPath(this._extensionUri, 'resources');

        // Get resources folder *webview* URI
        const resources_folder_webview_uri = webview.asWebviewUri(resources_folder_uri);

        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Explainer Recorder</title>
                <script src="${vscode.Uri.joinPath(resources_folder_webview_uri, 'explainer_recorder.js')}"></script>
            </head>
            <body>
                <hr>
                <button id="start-button">Start Capture</button>
                <button id="stop-button">Stop Capture</button>
                <button id="report-button">Report Details</button>
                <hr>
                <span id="details-text"> -no details- </span>
            </body>
            </html>`;
    }
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
function startMicrophoneRecording(webview: vscode.Webview, outputPath: fs.PathLike) {

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
            const fileUri = vscode.Uri.file(outputPath);
            webview.postMessage({ command: 'audioRecorded', fileUri: webview.asWebviewUri(fileUri) });
        });
    }
}
