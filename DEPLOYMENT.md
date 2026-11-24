# AgriSathi Deployment

This project can run in two useful ways:

1. Permanent local server on your own machine
2. Public deployment on a VPS or cloud server

## 1. Permanent local server

Use this when you want the app to keep running on your computer without opening the code each time.

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create your environment file:

```powershell
Copy-Item .env.example .env
```

Update `.env` with your real MySQL password and a real `SECRET_KEY`.

Initialize the database:

```powershell
.\.venv\Scripts\python.exe database.py
```

Run the app with a production server:

```powershell
.\.venv\Scripts\python.exe run_server.py
```

To make it start automatically on Windows after boot, create a Task Scheduler task that runs:

```powershell
c:\Users\Gaurav Dixit\OneDrive\Desktop\AgriSathi\.venv\Scripts\python.exe c:\Users\Gaurav Dixit\OneDrive\Desktop\AgriSathi\run_server.py
```

Set the task to:

- Run whether user is logged in or not
- Start at system startup
- Start in `c:\Users\Gaurav Dixit\OneDrive\Desktop\AgriSathi`

## 2. Deploy on a VPS

Best simple target: an Ubuntu server with Python, MySQL, and Nginx.

Copy the project to the server, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python database.py
python run_server.py
```

For permanent hosting on Linux, run `run_server.py` with `systemd` and place Nginx in front of it as a reverse proxy.

Suggested flow:

1. App runs on `127.0.0.1:5000`
2. Nginx listens on port `80` or `443`
3. Nginx forwards requests to the Python app
4. MySQL runs locally or on a managed database server

## Important note

This app uses large PyTorch model files. Make sure the server has enough disk space and RAM before deployment.
