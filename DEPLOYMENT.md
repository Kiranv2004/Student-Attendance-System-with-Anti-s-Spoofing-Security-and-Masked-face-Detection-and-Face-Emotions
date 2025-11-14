# 🚀 Deployment Guide

This guide covers multiple deployment options for the Student Attendance System.

## 📋 Prerequisites

- Python 3.9+
- MongoDB database (local or cloud)
- Git repository access
- Model files in `face_security/resources/`

---

## 🌐 Deployment Options

### Option 1: Railway (Recommended - Easy & Free Tier)

**Railway** is excellent for Python apps with MongoDB.

#### Steps:

1. **Sign up at [Railway.app](https://railway.app)**

2. **Create New Project:**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Connect your GitHub account
   - Select your repository

3. **Add MongoDB Service:**
   - Click "+ New"
   - Select "Database" → "MongoDB"
   - Railway will provide connection string

4. **Configure Environment Variables:**
   ```
   MONGODB_URI=<railway_mongodb_connection_string>
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your_email@gmail.com
   SMTP_PASSWORD=your_app_password
   FROM_EMAIL=your_email@gmail.com
   FROM_NAME=College Attendance System
   TEACHER_EMAILS=teacher1@example.com,teacher2@example.com
   ```

5. **Deploy:**
   - Railway auto-detects Python apps
   - Uses `Procfile` for startup
   - Deploys automatically on git push

6. **Access Your App:**
   - Railway provides a public URL
   - Example: `https://your-app.railway.app`

---

### Option 2: Render (Free Tier Available)

**Render** offers free tier with automatic deployments.

#### Steps:

1. **Sign up at [Render.com](https://render.com)**

2. **Create New Web Service:**
   - Connect GitHub repository
   - Select "Web Service"
   - Choose your repo

3. **Configure:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python fixed_integrated_attendance_system.py`
   - **Environment:** Python 3

4. **Add MongoDB:**
   - Create "MongoDB" service
   - Copy connection string

5. **Set Environment Variables:**
   - Same as Railway (see above)

6. **Deploy:**
   - Click "Create Web Service"
   - Render builds and deploys automatically

---

### Option 3: Heroku (Paid - $7/month minimum)

**Heroku** is reliable but requires paid dynos for production.

#### Steps:

1. **Install Heroku CLI:**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login:**
   ```bash
   heroku login
   ```

3. **Create App:**
   ```bash
   heroku create your-app-name
   ```

4. **Add MongoDB Addon:**
   ```bash
   heroku addons:create mongolab:sandbox
   ```

5. **Set Environment Variables:**
   ```bash
   heroku config:set SMTP_SERVER=smtp.gmail.com
   heroku config:set SMTP_PORT=587
   heroku config:set SMTP_USERNAME=your_email@gmail.com
   heroku config:set SMTP_PASSWORD=your_app_password
   heroku config:set FROM_EMAIL=your_email@gmail.com
   heroku config:set FROM_NAME="College Attendance System"
   heroku config:set TEACHER_EMAILS=teacher1@example.com,teacher2@example.com
   ```

6. **Deploy:**
   ```bash
   git push heroku main
   ```

7. **Open App:**
   ```bash
   heroku open
   ```

---

### Option 4: Docker Deployment

Deploy using Docker on any platform (AWS, DigitalOcean, etc.).

#### Build Docker Image:

```bash
docker build -t attendance-system .
```

#### Run Container:

```bash
docker run -d \
  -p 5000:5000 \
  -e MONGODB_URI=mongodb://localhost:27017/ \
  -e SMTP_SERVER=smtp.gmail.com \
  -e SMTP_USERNAME=your_email@gmail.com \
  -e SMTP_PASSWORD=your_app_password \
  --name attendance-system \
  attendance-system
```

#### Docker Compose (with MongoDB):

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/attendance_system
      - SMTP_SERVER=smtp.gmail.com
      - SMTP_USERNAME=your_email@gmail.com
      - SMTP_PASSWORD=your_app_password
    depends_on:
      - mongodb

volumes:
  mongodb_data:
```

Run:
```bash
docker-compose up -d
```

---

### Option 5: VPS (DigitalOcean, Linode, AWS EC2)

Deploy on a Virtual Private Server for full control.

#### Steps:

1. **Create VPS:**
   - Ubuntu 22.04 LTS
   - Minimum: 2GB RAM, 2 CPU cores
   - Recommended: 4GB RAM, 4 CPU cores

2. **SSH into Server:**
   ```bash
   ssh root@your-server-ip
   ```

3. **Install Dependencies:**
   ```bash
   # Update system
   apt update && apt upgrade -y
   
   # Install Python
   apt install python3.9 python3-pip python3-venv -y
   
   # Install MongoDB
   apt install mongodb -y
   systemctl start mongodb
   systemctl enable mongodb
   
   # Install Nginx (reverse proxy)
   apt install nginx -y
   ```

4. **Clone Repository:**
   ```bash
   cd /var/www
   git clone https://github.com/Kiranv2004/Student-Attendance-System-with-Anti-s-Spoofing-Security-and-Masked-face-Detection-and-Face-Emotions.git
   cd Student-Attendance-System-with-Anti-s-Spoofing-Security-and-Masked-face-Detection-and-Face-Emotions
   ```

5. **Setup Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

6. **Create Systemd Service:**
   Create `/etc/systemd/system/attendance-system.service`:
   ```ini
   [Unit]
   Description=Student Attendance System
   After=network.target mongodb.service

   [Service]
   User=www-data
   WorkingDirectory=/var/www/Student-Attendance-System-with-Anti-s-Spoofing-Security-and-Masked-face-Detection-and-Face-Emotions
   Environment="PATH=/var/www/Student-Attendance-System-with-Anti-s-Spoofing-Security-and-Masked-face-Detection-and-Face-Emotions/venv/bin"
   ExecStart=/var/www/Student-Attendance-System-with-Anti-s-Spoofing-Security-and-Masked-face-Detection-and-Face-Emotions/venv/bin/python fixed_integrated_attendance_system.py

   [Install]
   WantedBy=multi-user.target
   ```

7. **Start Service:**
   ```bash
   systemctl daemon-reload
   systemctl enable attendance-system
   systemctl start attendance-system
   ```

8. **Configure Nginx:**
   Create `/etc/nginx/sites-available/attendance-system`:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

   Enable site:
   ```bash
   ln -s /etc/nginx/sites-available/attendance-system /etc/nginx/sites-enabled/
   nginx -t
   systemctl reload nginx
   ```

---

## 🔧 Environment Variables

Set these in your deployment platform:

| Variable | Description | Example |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017/` |
| `SMTP_SERVER` | SMTP server address | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP port | `587` |
| `SMTP_USERNAME` | Email username | `your_email@gmail.com` |
| `SMTP_PASSWORD` | Email app password | `your_app_password` |
| `FROM_EMAIL` | Sender email | `your_email@gmail.com` |
| `FROM_NAME` | Sender name | `College Attendance System` |
| `TEACHER_EMAILS` | Teacher emails (comma-separated) | `teacher1@example.com,teacher2@example.com` |

---

## ⚠️ Important Notes

1. **Model Files:** Ensure `face_security/resources/` contains all model files
2. **MongoDB:** Use cloud MongoDB (MongoDB Atlas) for production
3. **Security:** Change Flask secret key in production
4. **HTTPS:** Use SSL certificates for production (Let's Encrypt is free)
5. **File Size:** Model files are large - consider Git LFS for version control

---

## 🐛 Troubleshooting

### Issue: "Module not found"
- Ensure all dependencies are in `requirements.txt`
- Check virtual environment is activated

### Issue: "MongoDB connection failed"
- Verify MongoDB is running
- Check connection string is correct
- Ensure MongoDB is accessible from deployment platform

### Issue: "Port already in use"
- Change port in `fixed_integrated_attendance_system.py`
- Or use environment variable: `PORT=8080`

---

## 📞 Support

For deployment issues, check:
- Platform-specific documentation
- Application logs
- MongoDB connection status
- Environment variables

---

**Recommended for Beginners:** Railway or Render (easiest setup)  
**Recommended for Production:** VPS with Nginx + Gunicorn (best performance)

