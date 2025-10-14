# Running the Chess GUI on WSL

The chess GUI requires a graphical display. Here are your options for running it on WSL:

## Option 1: Using Windows X Server (Recommended)

### Step 1: Install VcXsrv (Windows)

1. Download and install [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
2. Launch XLaunch
3. Select "Multiple windows" → Next
4. Select "Start no client" → Next
5. **Important**: Check "Disable access control" → Next
6. Click Finish

### Step 2: Set DISPLAY in WSL

```bash
# Add to your ~/.bashrc or run in terminal
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0

# Or for Windows 11 with WSLg (automatic)
export DISPLAY=:0
```

### Step 3: Install xcb dependencies

```bash
sudo apt update
sudo apt install -y libxcb-cursor0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0
```

### Step 4: Run the GUI

```bash
cd /home/fra/Uni/Thesis/main_repo/FedRL/chess-federated-learning
uv run python play_chess.py
```

## Option 2: Using WSLg (Windows 11 Only)

If you're on Windows 11, WSLg provides built-in GUI support:

```bash
# Install dependencies
sudo apt install -y libxcb-cursor0

# Run directly
cd /home/fra/Uni/Thesis/main_repo/FedRL/chess-federated-learning
uv run python play_chess.py
```

## Option 3: Remote Desktop (RDP)

```bash
# Install xrdp
sudo apt install -y xrdp xfce4

# Start xrdp
sudo service xrdp start

# Connect from Windows using Remote Desktop to localhost:3389
```

## Option 4: Run on Native Windows

If the above options don't work, you can copy the chess-federated-learning directory to Windows and run it natively:

```bash
# In WSL, copy to Windows
cp -r /home/fra/Uni/Thesis/main_repo/FedRL ~/../../mnt/c/Users/YourUsername/Desktop/

# Then in Windows PowerShell/CMD:
cd C:\Users\YourUsername\Desktop\FedRL\chess-federated-learning
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install PyQt6
python play_chess.py
```

## Troubleshooting

### "Could not load Qt platform plugin xcb"

Install missing dependencies:
```bash
sudo apt install -y libxcb-cursor0 libxcb-xinerama0 libxcb-icccm4
```

### "Cannot open display"

Check DISPLAY variable:
```bash
echo $DISPLAY
# Should show something like 192.168.x.x:0.0

# Test with a simple X app
sudo apt install x11-apps
xcalc  # Should open a calculator
```

### Black screen or "Failed to connect to socket"

1. Make sure VcXsrv is running on Windows
2. Disable Windows Firewall temporarily to test
3. Check "Disable access control" in XLaunch settings

## Quick Test

Test if your X server is working:

```bash
# Install test app
sudo apt install x11-apps

# Try to open a simple window
xeyes

# If xeyes opens, your X server is working!
```

## Performance Tips

- Use VcXsrv with "-ac" flag for better performance
- Close other X applications when playing chess
- Consider using native Windows if performance is poor
