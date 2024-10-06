// Imports
import * as vscode from 'vscode';

var line: number = 0;

// Activate Extension
export function activate(context: vscode.ExtensionContext) {
	context.subscriptions.push(

		// Register Command
		vscode.commands.registerCommand('code-explainer', () => {

			// Get resources folder URI
			const resources_folder_uri: vscode.Uri = vscode.Uri.joinPath(context.extensionUri, 'resources');

			// Create and show panel
			const panel = vscode.window.createWebviewPanel(
				'code-explainer',
				'Code Explainer',
				vscode.ViewColumn.Two,
				{
					enableScripts: true,
					localResourceRoots: [resources_folder_uri]
				}
			);

			// Get the active text editor
			const editor = vscode.window.activeTextEditor;

			// Get resources folder *webview* URI
			const resources_folder_webview_uri = panel.webview.asWebviewUri(resources_folder_uri);

			// Set webview HTML content
			panel.webview.html = getWebviewContent(resources_folder_webview_uri);

			// Handle messages from the webview
			panel.webview.onDidReceiveMessage(
				message => {
					switch (message.command) {
						case 'updatePosition':
							console.log(`Current playback position: ${message.currentTime} seconds`);
							line = Math.round(message.currentTime);
							if (editor) {
								const document = editor.document;
								const selection = editor.selection;

								// Get file name in active text editor
								const file_name = document.fileName;

								const highlight_range = new vscode.Selection(line, 0, line, 200);
								line = line + 1;
								const highlighted = editor.document.getText(highlight_range);
								editor.selection = highlight_range;
							}
							break;
						case 'audioEnded':
							break;
					}
				},
				undefined,
				context.subscriptions
			);

		})
	);
}

// Get webview HTML content
function getWebviewContent(resources_uri: vscode.Uri): string {
	return `
      <!DOCTYPE html>
      <html lang="en">
      <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Code Explainer</title>
          <script src="${vscode.Uri.joinPath(resources_uri, 'code_explainer.js')}"></script>
      </head>
      <body>
          <hr>
		  <audio id="audioPlayer" controls autoplay>
		    <source src="${vscode.Uri.joinPath(resources_uri, 'audio_debug.mp3')}" type="audio/mpeg">
		    Your browser does not support the audio element.
		  </audio>
          <hr>
	  </body>
      </html>
  `;
}

//FIN