/*
  RGB Server
    - Control the intensity using PWM of Red, Green, and Blue LEDs in response to serial commands
    - Respond to "color:value" pairs terminated with either newline or carriage return, e.g. "r:231\n" or "b:127\r"
*/
// Constants
#define R_PIN 3
#define G_PIN 5
#define B_PIN 6

// Globals
char    command_type;
char    command_value[3];
bool    command_valid = false;
int     command_index = 0;

// Setup
void setup() {
  // Initialize serial port
  Serial.begin(115200);
}

// Loop
void loop() {
  
  // Check for any incoming bytes
  if (Serial.available() > 0) {
      char new_char = Serial.read();
      
      // Check for command terminator
      if ((new_char == '\n') || (new_char == '\r'))
      {
        // Validate command
        if (command_index == 5)
        {
          command_valid = true;        
        }

        // Reset command index
        command_index = 0;
      }
      else
      {
        // Build command
        switch(command_index)
        {
          case 0:
            command_type = new_char;
            break;
          case 1:
            break;
          case 2:
            command_value[0] = new_char;
            break;
          case 3:
            command_value[1] = new_char;
            break;
          case 4:
            command_value[2] = new_char;
            break;
          default:
            break;
        }
        command_index += 1;
      }    
  }

  // If valid command...
  if (command_valid)
  {
    // Convert to numerical intensity (0-255)
    int intensity = atoi(command_value);
    
    // Report command feedback
    char command_feedback[10+11+10+3];
    sprintf(command_feedback, "Command: %c - Value: %i", command_type, intensity);
    Serial.println(command_feedback);

    // Respond to command
    if (command_type == 'r')
    {
      analogWrite(R_PIN, intensity);
    }
    if (command_type == 'g')
    {
      analogWrite(G_PIN, intensity);
    }
    if (command_type == 'b')
    {
      analogWrite(B_PIN, intensity);
    }
 
    // Reset for next command
    command_valid = false;
  }

  // Wait a bit
  delay(2);
}