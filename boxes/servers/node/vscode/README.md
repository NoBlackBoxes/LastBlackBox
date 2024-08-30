# Servers : Node : VSCode

VSCode is an "Electron" app. Electron is a framework that uses Chromium (open-source base of the Chrome browser) to develop and deliever cross-platform applications. Electron is built on top of Node.js, a server-side platform for running Javascript outside the browser.

## Development

Create a launch.json file in your .vscode folder with the following content.

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run LBB Extension",
            "type": "extensionHost",
            "request": "launch",
            "runtimeExecutable": "${execPath}",
            "args": [
                "--extensionDevelopmentPath=${workspaceFolder}/${relativeFileDirname}/.."
            ],
            "outFiles": [
                "${workspaceFolder}/${relativeFileDirname}/../out/**/*.js"
            ],
            "sourceMaps": true,
            "smartStep": true
        }
    ]
}
```