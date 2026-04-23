# ClassGuard AI - Classroom Behavior Monitor

ClassGuard AI is a real-time, browser-based monitoring system designed to track and detect forbidden objects in a classroom setting using a webcam. It uses TensorFlow.js and the `coco-ssd` pre-trained AI model to execute high-speed object detection **entirely locally in the browser**—meaning no recordings or images are ever sent to an external server.

Currently configured to flag restricted objects such as:
- Cell Phones
- Laptops
- Knives
- Cups, Bottles, and more.

---

## 🚀 Setup Instructions (For Complete Beginners)

If you have a brand-new computer and have never coded before, follow these steps exactly to get the project running!

### Step 1: Install Node.js
This project requires **Node.js** to run its local development server. 
1. Go to the official website: [https://nodejs.org/](https://nodejs.org/)
2. Download the **"LTS" (Long Term Support)** installer for your operating system (Windows/Mac).
3. Run the installer and click "Next" through the standard setup. (You can leave all the default settings as they are).
4. Once installation is complete, your computer will have Node.js and its package manager (`npm`) installed.

### Step 2: Open the Terminal in the Project Folder
You need to open a command line (Terminal) directly in the folder where this project is located.
- **On Windows**: Open the `AI project` folder in File Explorer. Click the address bar at the top, type `cmd`, and press **Enter**. This will open a black terminal window directly inside the project folder.
- **On Mac**: Open the folder, right-click, and select **"New Terminal at Folder"**.

### Step 3: Install the Project Dependencies
In the terminal window you just opened, type the following command and press **Enter**:
```bash
npm install
```
*Wait a few minutes.* This command reads the project configuration and automatically downloads the AI libraries (TensorFlow.js) and the Vite development tools needed to run the website.

### Step 4: Start the Website!
Once the installation finishes, type this command and press **Enter**:
```bash
npm run dev
```

You will see output in the terminal that looks like this:
``` text
  VITE v8.0.10  ready in 1254 ms

  ➜  Local:   http://localhost:5173/
```

### Step 5: Open it in your Web Browser
1. Open Google Chrome, Firefox, or Edge.
2. In the URL bar at the top, type `http://localhost:5173/` and hit enter.
3. **Important**: The website relies on checking the camera feed. Your browser will pop up a message asking for **Camera Permissions**. Click **Allow**.
4. Click the **Start Camera** button on the left sidebar.
5. The AI model will initialize (taking ~5-15 seconds depending on your PC) and then the live video feed will appear.

---

## 🛠 Troubleshooting

- **"Requested device not found" Error**: This happens if your computer does not have a physical webcam plugged in or turned on. You must have a working webcam for the live monitoring to start.
- **"npm is not recognized"**: This means Node.js from Step 1 did not install correctly or your PC needs a restart. Restart your computer and try Step 2 again. 
- **The screen is dark / model stuck on loading**: Ensure your internet connection is active on the very first load. The browser needs to make a one-time download of the AI mind ("model weights") from Google's servers before it can run.

## 💻 Tech Stack
- Frontend: HTML5, CSS3, Vanilla JS
- Build Tool: Vite
- AI/Machine Learning: `@tensorflow/tfjs` & `@tensorflow-models/coco-ssd`

