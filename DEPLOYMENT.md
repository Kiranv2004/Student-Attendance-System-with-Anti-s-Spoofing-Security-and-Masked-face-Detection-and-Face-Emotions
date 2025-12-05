# 🚀 Deployment Guide for Render.com

This guide will help you deploy the Student Attendance System to Render.com.

---

## 📋 Prerequisites

1. **Render.com Account**: Sign up at [render.com](https://render.com)
2. **MongoDB Atlas Account**: Free tier available at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
3. **GitHub Repository**: Push your code to GitHub (recommended)

---

## 🔧 Step 1: Set Up MongoDB Atlas (Cloud Database)

### 1.1 Create MongoDB Atlas Account
1. Go to [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Sign up for a free account
3. Create a new cluster (choose FREE tier)

### 1.2 Configure Database Access
1. Go to **Database Access** → **Add New Database User**
2. Create a username and password (save these!)
3. Set privileges to **Read and write to any database**

### 1.3 Configure Network Access
1. Go to **Network Access** → **Add IP Address**
2. Click **Allow Access from Anywhere** (0.0.0.0/0) for Render deployment
   - Or add Render's IP ranges if you prefer more security

### 1.4 Get Connection String
1. Go to **Database** → **Connect**
2. Choose **Connect your application**
3. Copy the connection string (looks like):
   ```
   mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
4. Add database name to the connection string:
   ```
   mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/attendance_system?retryWrites=true&w=majority
   ```

---

## 📦 Step 2: Prepare Your Code for Deployment

### 2.1 Push Code to GitHub
```bash
git init
git add .
git commit -m "Initial commit - Ready for Render deployment"
git branch -M main
git remote add origin https://github.com/yourusername/StudentAttendanceSystem.git
git push -u origin main
```

### 2.2 Verify Required Files
Ensure these files exist in your repository:
- ✅ `Procfile` - Tells Render how to start your app
- ✅ `requirements.txt` - Python dependencies
- ✅ `render.yaml` - Render configuration (optional but recommended)
- ✅ `fixed_integrated_attendance_system.py` - Main application file
- ✅ `templates/` - HTML templates directory
- ✅ `face_security/` - Anti-spoofing models directory

---

## 🌐 Step 3: Deploy to Render.com

### 3.1 Create New Web Service
1. Log in to [render.com](https://render.com)
2. Click **New +** → **Web Service**
3. Connect your GitHub repository
4. Select your repository: `StudentAttendanceSystem`

### 3.2 Configure Service Settings

**Basic Settings:**
- **Name**: `student-attendance-system` (or your preferred name)
- **Region**: Choose closest to your users
- **Branch**: `main` (or your default branch)
- **Root Directory**: Leave empty (or `./` if needed)
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn fixed_integrated_attendance_system:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120`

**OR use the `render.yaml` file** (recommended):
- Render will automatically detect `render.yaml` and use those settings

### 3.3 Set Environment Variables

Click **Environment** tab and add these variables:

#### Required Variables:
```
MONGODB_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/attendance_system?retryWrites=true&w=majority
SECRET_KEY=your-secret-key-here-generate-random-string
```

#### Email Configuration (Optional but Recommended):
```
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password
FROM_EMAIL=your_email@gmail.com
FROM_NAME=College Attendance System
TEACHER_EMAILS=teacher1@example.com,teacher2@example.com
```

#### Optional Variables:
```
PYTHON_VERSION=3.9.13
MONGODB_TIMEOUT=5000
FLASK_ENV=production
```

### 3.4 Generate Secret Key

Generate a secure secret key:
```python
import secrets
print(secrets.token_hex(32))
```

Or use online tool: https://randomkeygen.com/

---

## 🔐 Step 4: Gmail App Password Setup

If using Gmail for email notifications:

1. **Enable 2-Factor Authentication** on your Google account
2. Go to [Google App Passwords](https://myaccount.google.com/apppasswords)
3. Generate a new app password for "Mail"
4. Copy the 16-character password
5. Use this password in `SMTP_PASSWORD` (not your regular Gmail password)

---

## ✅ Step 5: Deploy and Verify

### 5.1 Start Deployment
1. Click **Create Web Service**
2. Render will:
   - Clone your repository
   - Install dependencies from `requirements.txt`
   - Build your application
   - Start the web service

### 5.2 Monitor Build Logs
- Watch the build logs for any errors
- Common issues:
  - Missing dependencies → Add to `requirements.txt`
  - MongoDB connection errors → Check `MONGODB_URI`
  - Port binding errors → Ensure using `$PORT` variable

### 5.3 Verify Deployment
1. Once deployed, you'll get a URL like: `https://student-attendance-system.onrender.com`
2. Visit the URL to verify the app is running
3. Test registration and attendance features

---

## 🐛 Troubleshooting

### Issue: Build Fails - "dlib installation failed"
**Solution**: 
- dlib requires system dependencies
- Add `buildpack` or use pre-built wheels
- Consider using `dlib-binary` package instead

### Issue: "MongoDB connection failed"
**Solution**:
- Verify `MONGODB_URI` is correct
- Check MongoDB Atlas network access (allow 0.0.0.0/0)
- Ensure database user has read/write permissions
- Check connection string includes database name

### Issue: "Module not found" errors
**Solution**:
- Ensure all dependencies are in `requirements.txt`
- Check `face_security` module files are committed to Git
- Verify file paths are correct

### Issue: "Application crashed" or "502 Bad Gateway"
**Solution**:
- Check logs in Render dashboard
- Verify `Procfile` start command is correct
- Ensure app binds to `0.0.0.0:$PORT`
- Check timeout settings (increase if needed)

### Issue: "Out of memory" errors
**Solution**:
- Upgrade Render plan (free tier has 512MB RAM limit)
- Reduce number of workers in gunicorn
- Optimize model loading (lazy loading)

### Issue: Email not sending
**Solution**:
- Verify SMTP credentials are correct
- Check Gmail App Password (not regular password)
- Verify environment variables are set correctly
- Check Render logs for SMTP errors

---

## 📊 Render.com Free Tier Limitations

- **512MB RAM**: May be insufficient for ML models
- **750 hours/month**: ~31 days of continuous uptime
- **Sleeps after 15 minutes**: Free tier services sleep after inactivity
- **Cold starts**: First request after sleep takes 30-60 seconds

**Recommendations**:
- Consider upgrading to **Starter Plan** ($7/month) for:
  - 512MB RAM (same)
  - Always-on (no sleeping)
  - Better performance

---

## 🔄 Updating Your Deployment

### Automatic Deployments
- Render automatically deploys when you push to the connected branch
- Monitor build logs in Render dashboard

### Manual Deployments
1. Go to Render dashboard
2. Click **Manual Deploy** → **Deploy latest commit**

### Rollback
1. Go to **Events** tab
2. Find previous successful deployment
3. Click **Redeploy**

---

## 📝 Post-Deployment Checklist

- [ ] MongoDB Atlas cluster is running
- [ ] Database connection string is correct
- [ ] Environment variables are set
- [ ] Application starts without errors
- [ ] Registration page works
- [ ] Attendance capture works
- [ ] Email notifications work
- [ ] Analytics page loads correctly
- [ ] Face recognition models load successfully

---

## 🔒 Security Best Practices

1. **Never commit secrets** to Git
   - Use environment variables for all sensitive data
   - Add `.env` to `.gitignore`

2. **Use strong SECRET_KEY**
   - Generate random 32+ character string
   - Never reuse keys across environments

3. **MongoDB Security**
   - Use strong database passwords
   - Restrict network access if possible
   - Regularly rotate credentials

4. **HTTPS**
   - Render provides HTTPS automatically
   - No additional configuration needed

---

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [MongoDB Atlas Documentation](https://docs.atlas.mongodb.com/)
- [Gunicorn Documentation](https://gunicorn.org/)
- [Flask Deployment Guide](https://flask.palletsprojects.com/en/2.3.x/deploying/)

---

## 🆘 Support

If you encounter issues:
1. Check Render build logs
2. Check Render runtime logs
3. Verify all environment variables
4. Test MongoDB connection separately
5. Review this deployment guide

---

**Happy Deploying! 🚀**

