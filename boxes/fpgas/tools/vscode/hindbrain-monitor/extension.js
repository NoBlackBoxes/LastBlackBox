// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require('vscode');
var path = require("path");
const fs = require('fs');

const { exec } = require('child_process');

// Require Serial package
const { SerialPort } = require('serialport')

async function listFTDIPorts() {
	await SerialPort.list().then((ports, err) => {
		for (const port of ports) {
			if (port.manufacturer == 'FTDI') {
				console.log(port);
				console.log(port.pnpId);
				console.log(port.path);
			}
		}
	})
}

async function select() {
	let workspace_folder = vscode.workspace.workspaceFolders[0].uri.path;
	projects_folder = `${workspace_folder}/boxes/fpgas/hdl/verilog`;
	const project_list = fs.readdirSync(projects_folder);
	const result = await vscode.window.showQuickPick(project_list);
	project_folder = `${workspace_folder}/boxes/fpgas/hdl/verilog/${result}`;
	return (project_folder);
}

async function upload(project_folder) {
	project_binary = `${project_folder}/synthesis/bin/hardware.bin`;
	// Use "iceprog" to program the flash
	command = `iceprog -d i:0x0403:0x6010:0 ${project_binary}`
	console.log(`Command: ${project_binary}`)
	exec(command, (error, stdout, stderr) => {
		console.log(`stdout:\n${stderr}`);
	});
	return;
}

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('Congratulations, your extension "Hindbrain Monitor" is now active!');

	// The command has been defined in the package.json file
	// Now provide the implementation of the command with  registerCommand
	// The commandId parameter must match the command field in package.json
	let disposable = vscode.commands.registerCommand('hindbrain-monitor', async () => {
		// The code you place here will be executed every time your command is executed

		listFTDIPorts(); // Check that board is connected

		// Select a FPGA project to upload
		let project_folder = await select();

		// Upload core
		upload(project_folder);

		// Create serial monitor (for graphing and display)

		//// baudRate is specific to my project 
		//const myPort = new SerialPort({path: "/dev/ttyUSB1", baudRate: 9600});
		//var byte_buffer = new Uint8Array(300);
		//for (let i = 0; i < 300; i++) {
		//	byte_buffer[i] = i % 2;
		//}
		//myPort.write(Buffer.from(byte_buffer));
		//myPort.close(function (err) {
		//	console.log('port closed', err);
		//});
		//myPort.destroy(function (err) {
		//	console.log('port destroy', err);
		//})

		// Display a message box to the user
		vscode.window.showInformationMessage('Hello World from Hindbrain Monitor');
	});

	context.subscriptions.push(disposable);
}

// This method is called when your extension is deactivated
function deactivate() { }

module.exports = {
	activate,
	deactivate
}
