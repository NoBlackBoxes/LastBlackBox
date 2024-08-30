// Update plot
function update(interval)
{
    // Set update interval (milliseconds)
    setInterval(draw, interval);
}

// Random canvas line plot
function draw() {
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");
    context.canvas.width  = window.innerWidth;
    context.canvas.height = window.innerHeight/2;
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

    // Generate random values and plot
    context.beginPath();
    context.moveTo(0, shift_y);
    for (let index = 0; index < context.canvas.width; index++) {
        random = random_integer(-100, 100);
        context.lineTo(index, random+shift_y);
    }
    context.lineWidth = 1;
    context.strokeStyle = "yellow";
    context.stroke();
}

// Generate random integer
function random_integer(min, max) {
    const minCeiled = Math.ceil(min);
    const maxFloored = Math.floor(max);
    return Math.floor(Math.random() * (maxFloored - minCeiled + 1) + minCeiled);
}
