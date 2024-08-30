let serialData = []; // Array to store received serial data

// Listen for messages from the VS Code extension
window.addEventListener('message', event => {
    const message = event.data;
    switch (message.command) {
        case 'newData':
            const buffer = message.data
            const floatArray = buffer.map(byte => 200.0*((byte / 255) - 0.5));
            floatArray.forEach(value => {
                serialData.push(value);
                
                // Limit the array size to the canvas width
                if (serialData.length > window.innerWidth) {
                    serialData.shift(); // Remove the oldest data point
                }
            });
            break;
    }
});

// Update plot
function update(interval) {

    // Set update interval (milliseconds)
    setInterval(draw, interval);
}

// Draw canvas line plot
function draw() {
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");
    context.canvas.width  = window.innerWidth;
    context.canvas.height = window.innerHeight / 2;
    let shift_y = canvas.height / 2.0;

    // Clear canvas
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Draw Axis
    context.beginPath();
    context.moveTo(0, shift_y);
    context.lineTo(context.canvas.width, shift_y);
    context.lineWidth = 1;
    context.strokeStyle = "red";
    context.stroke();

    // Plot the serial data
    context.beginPath();
    context.moveTo(0, shift_y);
    for (let index = 0; index < serialData.length; index++) {
        context.lineTo(index, shift_y - serialData[index]);
    }
    context.lineWidth = 1;
    context.strokeStyle = "yellow";
    context.stroke();
}