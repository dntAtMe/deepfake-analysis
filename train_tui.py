"""Terminal User Interface for model training configuration."""
import curses
import sys
from pathlib import Path

class TrainingTUI:
    def __init__(self):
        print("Initializing TrainingTUI...")  # Debug output
        self.screen = None
        self.current_row = 0
        
        # Available options
        self.models = {
            "SpecRNet": "specrnet",
            "ResNet2": "resnet",
            "BaselineCNN": "cnn"
        }
        
        # Training configurations
        self.input_types = ["spectrogram", "mel", "mfcc", "lfcc"]
        self.batch_sizes = ["8", "16", "32", "64", "128"]
        self.learning_rates = ["0.1", "0.01", "0.001", "0.0001"]
        self.epochs = ["10", "30", "50", "100", "200"]
        
        print("Initialization complete")  # Debug output
    
    def _draw_menu(self, title: str, options: list) -> int:
        """Draw menu and get selection."""
        try:
            current_selection = 0
            while True:
                # Clear screen
                self.screen.clear()
                
                # Draw title
                self.screen.addstr(0, 0, f"=== {title} ===")
                
                # Draw options
                for idx, option in enumerate(options):
                    style = curses.A_REVERSE if idx == current_selection else curses.A_NORMAL
                    self.screen.addstr(idx + 2, 2, f"{option}", style)
                
                # Draw instructions
                self.screen.addstr(len(options) + 3, 0, "Use UP/DOWN arrows and ENTER to select")
                
                # Refresh screen
                self.screen.refresh()
                
                # Get input
                key = self.screen.getch()
                
                if key == curses.KEY_UP and current_selection > 0:
                    current_selection -= 1
                elif key == curses.KEY_DOWN and current_selection < len(options) - 1:
                    current_selection += 1
                elif key == ord('\n'):
                    return current_selection
                
        except Exception as e:
            self.screen.addstr(0, 0, f"Error in _draw_menu: {str(e)}")
            self.screen.refresh()
            self.screen.getch()
            return 0
    
    def main(self, stdscr):
        """Main TUI loop."""
        try:
            print("Starting main loop...")  # Debug output
            self.screen = stdscr
            
            # Setup screen
            curses.curs_set(0)  # Hide cursor
            self.screen.keypad(1)  # Enable keypad
            
            # Select model
            model_idx = self._draw_menu("Select Model", list(self.models.keys()))
            model_name = list(self.models.values())[model_idx]
            
            # Select input type
            input_idx = self._draw_menu("Select Input Type", self.input_types)
            input_type = self.input_types[input_idx]
            
            # Select batch size
            batch_idx = self._draw_menu("Select Batch Size", self.batch_sizes)
            batch_size = self.batch_sizes[batch_idx]
            
            # Select learning rate
            lr_idx = self._draw_menu("Select Learning Rate", self.learning_rates)
            learning_rate = self.learning_rates[lr_idx]
            
            # Select epochs
            epoch_idx = self._draw_menu("Select Number of Epochs", self.epochs)
            epochs = self.epochs[epoch_idx]
            
            # Show confirmation
            self.screen.clear()
            self.screen.addstr(0, 0, "=== Training Configuration ===")
            self.screen.addstr(2, 0, f"Model: {model_name}")
            self.screen.addstr(3, 0, f"Input Type: {input_type}")
            self.screen.addstr(4, 0, f"Batch Size: {batch_size}")
            self.screen.addstr(5, 0, f"Learning Rate: {learning_rate}")
            self.screen.addstr(6, 0, f"Epochs: {epochs}")
            self.screen.addstr(8, 0, "Start training? (y/n)")
            self.screen.refresh()
            
            # Get confirmation
            while True:
                key = self.screen.getch()
                if key in [ord('y'), ord('Y')]:
                    # Exit curses before launching training
                    curses.endwin()
                    
                    # Launch training script
                    cmd = [
                        sys.executable, "train.py",
                        "--model", model_name,
                        "--input-type", input_type,
                        "--batch-size", batch_size,
                        "--epochs", epochs,
                        "--lr", learning_rate
                    ]
                    print(f"Launching: {' '.join(cmd)}")
                    import subprocess
                    subprocess.run(cmd)
                    break
                    
                elif key in [ord('n'), ord('N')]:
                    break
            
        except Exception as e:
            curses.endwin()
            print(f"Error in main: {str(e)}")
            import traceback
            traceback.print_exc()

def run():
    """Run the TUI with proper error handling."""
    print("Starting TUI...")  # Debug output
    try:
        curses.wrapper(TrainingTUI().main)
    except Exception as e:
        print(f"Failed to start TUI: {str(e)}")
        import traceback
        traceback.print_exc()
    print("TUI finished")  # Debug output

if __name__ == "__main__":
    run()
