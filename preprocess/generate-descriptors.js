const faceapi = require("face-api.js");
const canvas = require("canvas");
const fetch = require("node-fetch");
const fs = require("fs");
const path = require("path");

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData, fetch });

const MODEL_PATH = path.join(__dirname, "models");
const DATASET_PATH = path.join(__dirname, "dataset", "images");
const OUTPUT = path.join(__dirname, "descriptors.json");

async function loadModels() {
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH);
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH);
}

async function run() {
  await loadModels();
  console.log("[INFO] Models loaded");

  const people = fs.readdirSync(DATASET_PATH);
  let labeledDescriptors = [];

  for (const person of people) {
    const personPath = path.join(DATASET_PATH, person);
    const files = fs.readdirSync(personPath);
    let descriptors = [];

    for (const file of files) {
      const imgPath = path.join(personPath, file);
      const img = await canvas.loadImage(imgPath);
      const detection = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection) {
        descriptors.push(detection.descriptor);
        console.log(`[INFO] Processed ${file} for ${person}`);
      } else {
        console.log(`[WARN] No face found in ${file}`);
      }
    }

    if (descriptors.length > 0) {
      labeledDescriptors.push({
        label: person,
        descriptors: descriptors.map((d) => Array.from(d)),
      });
    }
  }

  fs.writeFileSync(OUTPUT, JSON.stringify(labeledDescriptors, null, 2));
  console.log("[INFO] Saved descriptors.json");
}

run();
