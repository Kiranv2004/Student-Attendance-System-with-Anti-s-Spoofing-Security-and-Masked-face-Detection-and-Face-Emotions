# 🚀 Render.com Deployment Guide

This guide will help you deploy the Student Attendance System to Render.com.

## 📋 Prerequisites

1. **Render.com Account**: Sign up at [render.com](https://render.com)
2. **MongoDB Atlas Account**: Free tier available at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
3. **GitHub Repository**: Push your code to GitHub (or use Render's Git integration)

## 🔧 Step-by-Step Deployment

### Step 1: Set Up MongoDB Atlas

1. **Create MongoDB Atlas Account**
   - Go to https://www.mongodb.com/cloud/atlas
   - Sign up for free account
   - Create a new cluster (Free tier M0 is sufficient)

2. **Configure Database Access**
   - Go to "Database Access" → "Add New Database User"
   - Create a username and password (save these!)
   - Set privileges to "Read and write to any database"

3. **Configure Network Access**
   - Go to "Network Access" → "Add IP Address"
   - Click "Allow Access from Anywhere" (0.0.0.0/0) for Render deployment

4. **Get Connection String**
   - Go to "Database" → Click "Connect" on your cluster
   - Choose "Connect your application"
   - Copy the connection string (looks like: `mongodb+srv://username:password@cluster.mongodb.net/`)
   - Replace `<password>` with your actual password
   - Add database name: `mongodb+srv://username:password@cluster.mongodb.net/attendance_system`

### Step 2: Deploy to Render

1. **Create New Web Service**
   - Log in to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository (or use Render's Git)

2. **Configure Service Settings**
   - **Name**: `student-attendance-system` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python fixed_integrated_attendance_system.py`
   - **Plan**: Choose "Free" or "Starter" plan

3. **Set Environment Variables**
   Click "Environment" tab and add these variables:

   ```
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/attendance_system
   PORT=10000
   FLASK_ENV=production
   SECRET_KEY=your-secret-key-here-generate-random-string
   
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   FROM_EMAIL=your-email@gmail.com
   FROM_NAME=College Attendance System
   TEACHER_EMAILS=teacher1@example.com,teacher2@example.com
   ```

   **Important Notes:**
   - Replace `MONGODB_URI` with your MongoDB Atlas connection string
   - `PORT` is automatically set by Render (you can leave it or set to 10000)
   - `SECRET_KEY`: Generate a random string (e.g., use Python: `import secrets; print(secrets.token_hex(32))`)
   - For Gmail SMTP: Use App Password, not regular password
     - Enable 2-Factor Authentication
     - Generate App Password: https://myaccount.google.com/apppasswords

4. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy your application
   - Wait for deployment to complete (5-10 minutes)

### Step 3: Verify Deployment

1. **Check Build Logs**
   - Monitor the build logs for any errors
   - Common issues:
     - Missing dependencies → Check `requirements.txt`
     - MongoDB connection → Verify `MONGODB_URI`
     - Port issues → Render sets PORT automatically

2. **Test Application**
   - Once deployed, visit your Render URL (e.g., `https://student-attendance-system.onrender.com`)
   - Test registration, attendance capture, and analytics

## 📝 Environment Variables Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `MONGODB_URI` | ✅ Yes | MongoDB Atlas connection string | `mongodb+srv://user:pass@cluster.mongodb.net/attendance_system` |
| `PORT` | ✅ Yes | Server port (auto-set by Render) | `10000` |
| `FLASK_ENV` | ✅ Yes | Environment mode | `production` |
| `SECRET_KEY` | ✅ Yes | Flask secret key | Random 32+ character string |
| `SMTP_SERVER` | ⚠️ Optional | SMTP server for emails | `smtp.gmail.com` |
| `SMTP_PORT` | ⚠️ Optional | SMTP port | `587` |
| `SMTP_USERNAME` | ⚠️ Optional | Email username | `your-email@gmail.com` |
| `SMTP_PASSWORD` | ⚠️ Optional | Email app password | `app-password-here` |
| `FROM_EMAIL` | ⚠️ Optional | Sender email | `your-email@gmail.com` |
| `FROM_NAME` | ⚠️ Optional | Sender name | `College Attendance System` |
| `TEACHER_EMAILS` | ⚠️ Optional | Teacher emails (comma-separated) | `teacher1@example.com,teacher2@example.com` |

## 🔍 Troubleshooting

### Issue: Build Fails
**Solution:**
- Check `requirements.txt` for all dependencies
- Verify Python version compatibility (3.9+)
- Check build logs for specific error messages

### Issue: MongoDB Connection Failed
**Solution:**
- Verify `MONGODB_URI` is correct
- Check MongoDB Atlas Network Access (allow 0.0.0.0/0)
- Verify database user credentials
- Ensure connection string includes database name: `/attendance_system`

### Issue: Application Crashes on Start
**Solution:**
- Check Render logs for error messages
- Verify all required environment variables are set
- Ensure `PORT` environment variable is set (Render sets this automatically)
- Check if face_security models are accessible (may need to upload to cloud storage)

### Issue: Email Not Sending
**Solution:**
- Verify SMTP credentials are correct
- For Gmail, use App Password (not regular password)
- Check SMTP server allows connections from Render IPs
- Verify `FROM_EMAIL` matches `SMTP_USERNAME` for Gmail

### Issue: Face Recognition Not Working
**Solution:**
- Ensure `face_security` directory and model files are in repository
- Check file paths are correct (relative paths work better)
- Verify model files are not too large (Render has size limits)

## 📦 File Structure for Render

Ensure these files are in your repository:

```
StudentAttendanceSystem/
├── fixed_integrated_attendance_system.py  # Main application
├── requirements.txt                        # Python dependencies
├── Procfile                                # Process file for Render
├── render.yaml                             # Render configuration (optional)
├── .runtime.txt                           # Python version (optional)
├── face_security/                         # Anti-spoofing models
│   └── resources/
│       ├── anti_spoof_models/
│       └── detection_model/
├── templates/                              # HTML templates
│   ├── dashboard.html
│   ├── registration.html
│   ├── attendance.html
│   └── analytics.html
└── README.md
```

## 🎯 Post-Deployment Checklist

- [ ] MongoDB Atlas cluster is running
- [ ] Database connection string is correct
- [ ] All environment variables are set
- [ ] Application builds successfully
- [ ] Web interface is accessible
- [ ] Student registration works
- [ ] Attendance capture works
- [ ] Email notifications work (if configured)
- [ ] Analytics page loads correctly

## 💡 Tips

1. **Free Tier Limitations:**
   - Render free tier spins down after 15 minutes of inactivity
   - First request after spin-down takes ~30 seconds to wake up
   - Consider upgrading to Starter plan ($7/month) for always-on service

2. **Database:**
   - MongoDB Atlas free tier: 512MB storage
   - Sufficient for small to medium deployments
   - Monitor usage in Atlas dashboard

3. **Performance:**
   - Face recognition models are large (~100MB+)
   - First load may be slow
   - Consider using CDN for static assets

4. **Security:**
   - Never commit `.env` files or secrets to Git
   - Use Render's environment variables for all secrets
   - Rotate `SECRET_KEY` regularly

## 📞 Support

If you encounter issues:
1. Check Render logs: Dashboard → Your Service → Logs
2. Check MongoDB Atlas logs: Atlas Dashboard → Logs
3. Review application logs in Render dashboard
4. Verify all environment variables are set correctly

---

**Happy Deploying! 🚀**

