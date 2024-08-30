// Imports
import * as vscode from 'vscode';

// Activate Extension
export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(

    // Register Command
    vscode.commands.registerCommand('html-canvas', () => {

      // Get resources folder URI
      var resources_folder_uri: vscode.Uri = vscode.Uri.joinPath(context.extensionUri, 'resources');

      // Create and show panel
      const panel = vscode.window.createWebviewPanel(
        'html-canvas',
        'HTML Canvas',
        vscode.ViewColumn.Two,
        {
          // Enable scripts and access to resources folder inside the webview
          enableScripts: true,
          localResourceRoots: [resources_folder_uri]
        }
      );

      // Get resources folder *webview* URI
      const resources_folder_webview_uri = panel.webview.asWebviewUri(resources_folder_uri);

      // Set webview HTML content
      panel.webview.html = getWebviewContent(resources_folder_webview_uri);
    })
  );
}

// Get webview HTML content
function getWebviewContent(resources_uri: vscode.Uri) {
  let html: string = ``;
  html +=`
    <!DOCTYPE html>
      <html lang="en">
      <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>HTML Canvas</title>
          <script src="${vscode.Uri.joinPath(resources_uri, 'random_plot.js')}"></script>
      </head>
      <body onload="update()">
        <hr>
        <canvas id="canvas"></canvas>
        <hr>
      </body>
      </html>
    `;
  return html;
}
