// Imports
import * as vscode from 'vscode';
import * as SerialPort from 'serialport';
import { ByteLengthParser } from '@serialport/parser-byte-length';

// Activate Extension
export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(

    // Register Command
    vscode.commands.registerCommand('serial-plotter', () => {

      // Get resources folder URI
      const resources_folder_uri: vscode.Uri = vscode.Uri.joinPath(context.extensionUri, 'resources');

      // Create and show panel
      const panel = vscode.window.createWebviewPanel(
        'serial-plotter',
        'Serial Plotter',
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

      // Open the serial port
      openSerialPort(panel.webview);
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
          <title>Serial Plotter</title>
          <script src="${vscode.Uri.joinPath(resources_uri, 'serial_plot.js')}"></script>
      </head>
      <body onload="update(100)">
          <hr>
          <canvas id="canvas"></canvas>
          <hr>
      </body>
      </html>
  `;
}

// Function to open serial port
function openSerialPort(webview: vscode.Webview) {
  // Debug
  //const portPath = '/dev/pts/7';  // Hard-coded serial port path
  //const baudRate = 9600;  // Hard-coded baud rate

  // Live
  const portPath = '/dev/ttyUSB1';  // Hard-coded serial port path
  const baudRate = 115200;  // Hard-coded baud rate

  // Open port and specify data parser
  const port = new SerialPort.SerialPort({ path: portPath, baudRate: baudRate });
  const parser = port.pipe(new ByteLengthParser({ length: 8 }));

  // On "data" callback
  parser.on('data', (data: Buffer) => {
    // Debug: Log received data
    // console.log('Data received:', data);
    const byteArray = Array.from(data);
    webview.postMessage({ command: 'newData', data: byteArray });
  });

  // On "error" callback
  port.on('error', (err: Error) => {
    // Debug: Log errors
    console.error('Error:', err.message);
    webview.postMessage({ command: 'error', error: err.message });
  });

  // On "open" callback
  port.on('open', () => {
    // Debug: Log when the port is successfully opened
    console.log(`Serial port ${portPath} is open.`);
    webview.postMessage({ command: 'status', status: `Port ${portPath} is open` });
  });
}
