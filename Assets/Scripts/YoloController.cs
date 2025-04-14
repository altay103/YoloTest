using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using TensorFlowLite;
using UnityEngine.UI;
using TMPro;

public class YoloController : MonoBehaviour
{
    [Header("Model Settings")]
    [SerializeField, FilePopup("*.tflite")]
    private string modelFile = "yolov4-416-fp32.tflite";

    [Header("Detection Settings")]
    [Range(0f, 1f)]
    public float scoreThreshold = 0.5f;
    [Range(0f, 1f)]
    public float iouThreshold = 0.5f;

    [Header("References")]
    public RawImage cameraView;
    public TextMeshProUGUI frameRateText;
    public TextMeshProUGUI detectionCountText;

    private Interpreter interpreter;
    private float[,,,] inputTensor;
    private float[,,] output0;
    private float[,,] output1;

    private WebCamTexture webCamTexture;
    private Texture2D inputTexture;
    private Texture2D resizedTexture;

    private int inputWidth = 416;
    private int inputHeight = 416;

    private List<Detection> detections = new List<Detection>();
    private float fpsUpdateInterval = 0.5f;
    private float fpsAccumulator = 0f;
    private int fpsFrames = 0;
    private float fpsTimeLeft;

    private bool isProcessing = false;

    private List<Rect> boundingBoxPositions = new List<Rect>();

    // COCO veri setindeki 80 sınıf için etiketler
    private readonly string[] labels = new string[]
    {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    private void Start()
    {
        try
        {
            string modelPath = System.IO.Path.Combine(Application.streamingAssetsPath, modelFile);
            Debug.Log($"Model path: {modelPath}");
            if (!System.IO.File.Exists(modelPath))
            {
                Debug.LogError($"Model file does not exist at {modelPath}");
                return;
            }
            else
            {
                Debug.Log($"Model file exists at {modelPath}");
            }
            var options = new InterpreterOptions();
            options.threads = SystemInfo.processorCount;

            interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
            interpreter.AllocateTensors();

            // Çıktı tensorlarını başlangıçta oluştur
            var outputInfo0 = interpreter.GetOutputTensorInfo(0);
            var outputInfo1 = interpreter.GetOutputTensorInfo(1);
            output0 = new float[outputInfo0.shape[0], outputInfo0.shape[1], outputInfo0.shape[2]];
            output1 = new float[outputInfo1.shape[0], outputInfo1.shape[1], outputInfo1.shape[2]];

            LogModelInfo();

            InitCamera();

            inputTensor = new float[1, inputHeight, inputWidth, 3];

            fpsTimeLeft = fpsUpdateInterval;
        }
        catch (Exception e)
        {
            Debug.LogError($"Initialization error: {e.Message}\n{e.StackTrace}");
        }
    }

    private void LogModelInfo()
    {
        var inputInfo = interpreter.GetInputTensorInfo(0);
        Debug.Log($"Input shape: {string.Join(", ", inputInfo.shape)}");
        Debug.Log($"Input type: {inputInfo.type}");

        int outputCount = interpreter.GetOutputTensorCount();
        Debug.Log($"Model has {outputCount} outputs");

        for (int i = 0; i < outputCount; i++)
        {
            var info = interpreter.GetOutputTensorInfo(i);
            Debug.Log($"Output {i}: shape={string.Join(",", info.shape)}, type={info.type}");
        }
    }

    private void InitCamera()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.LogError("No camera found");
            return;
        }

        webCamTexture = new WebCamTexture(devices[0].name, inputWidth, inputHeight, 30);
        cameraView.texture = webCamTexture;
        webCamTexture.Play();
    }

    private void Update()
    {
        if (webCamTexture == null || !webCamTexture.isPlaying) return;

        UpdateFPS();

        if (!isProcessing)
        {
            StartCoroutine(ProcessFrame());
        }
    }

    private IEnumerator ProcessFrame()
    {
        isProcessing = true;
        yield return new WaitForEndOfFrame();

        ProcessInputTexture();
        RunInference();
        ProcessOutput();

        isProcessing = false;
    }

    private void ProcessInputTexture()
    {
        if (inputTexture == null || inputTexture.width != webCamTexture.width || inputTexture.height != webCamTexture.height)
        {
            inputTexture = new Texture2D(webCamTexture.width, webCamTexture.height, TextureFormat.RGB24, false);
        }
        if (resizedTexture == null)
        {
            resizedTexture = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);
        }

        inputTexture.SetPixels32(webCamTexture.GetPixels32());
        inputTexture.Apply();

        RenderTexture rt = RenderTexture.GetTemporary(inputWidth, inputHeight);
        Graphics.Blit(inputTexture, rt);
        RenderTexture.active = rt;

        resizedTexture.ReadPixels(new Rect(0, 0, inputWidth, inputHeight), 0, 0);
        resizedTexture.Apply();

        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        Color32[] pixels = resizedTexture.GetPixels32();
        for (int y = 0; y < inputHeight; y++)
        {
            for (int x = 0; x < inputWidth; x++)
            {
                Color32 color = pixels[y * inputWidth + x];
                inputTensor[0, y, x, 0] = color.r / 255f;
                inputTensor[0, y, x, 1] = color.g / 255f;
                inputTensor[0, y, x, 2] = color.b / 255f;
            }
        }
    }

    private void RunInference()
    {
        try
        {
            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }
        catch (Exception e)
        {
            Debug.LogError($"Inference error: {e.Message}");
        }
    }

    private void ProcessOutput()
    {
        detections.Clear();
        boundingBoxPositions.Clear();

        int numDetections = output0.GetLength(1);
        int numClasses = output1.GetLength(2);

        for (int i = 0; i < numDetections; i++)
        {
            float x = output0[0, i, 0];
            float y = output0[0, i, 1];
            float w = output0[0, i, 2];
            float h = output0[0, i, 3];

            // Koordinatları normalizasyon kontrolü
            x = Mathf.Clamp01(x);
            y = Mathf.Clamp01(y);
            w = Mathf.Clamp01(w);
            h = Mathf.Clamp01(h);

            float maxProb = 0;
            int classId = -1;

            for (int j = 0; j < numClasses; j++)
            {
                float prob = output1[0, i, j];
                if (prob > maxProb)
                {
                    maxProb = prob;
                    classId = j;
                }
            }

            if (maxProb >= scoreThreshold)
            {
                Rect rect = new Rect(x - w / 2, y - h / 2, w, h);
                detections.Add(new Detection
                {
                    rect = rect,
                    classId = classId,
                    score = maxProb
                });
            }
        }

        ApplyNonMaxSuppression();
        VisualizeDetections();
    }

    private void ApplyNonMaxSuppression()
    {
        for (int i = 0; i < detections.Count; i++)
        {
            for (int j = i + 1; j < detections.Count; j++)
            {
                if (CalculateIOU(detections[i].rect, detections[j].rect) > iouThreshold)
                {
                    if (detections[i].score > detections[j].score)
                    {
                        detections.RemoveAt(j);
                        j--;
                    }
                    else
                    {
                        detections.RemoveAt(i);
                        i--;
                        break;
                    }
                }
            }
        }
    }

    private float CalculateIOU(Rect a, Rect b)
    {
        float x1 = Mathf.Max(a.x, b.x);
        float y1 = Mathf.Max(a.y, b.y);
        float x2 = Mathf.Min(a.x + a.width, b.x + b.width);
        float y2 = Mathf.Min(a.y + a.height, b.y + b.height);

        float intersection = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
        float union = a.width * a.height + b.width * b.height - intersection;

        return intersection / union;
    }

    private void VisualizeDetections()
    {
        Debug.Log($"Detected {detections.Count} objects");

        foreach (var detection in detections)
        {
            float x = detection.rect.x * inputTexture.width;
            float y = detection.rect.y * inputTexture.height;
            float w = detection.rect.width * inputTexture.width;
            float h = detection.rect.height * inputTexture.height;

            DrawBoundingBox(x, y, w, h);

            string className = detection.classId < labels.Length ? labels[detection.classId] : "unknown";
            Debug.Log($"Detection: Class {className}, Score: {detection.score:F2}");
            detectionCountText.text = $"Detected: {className} ({detection.score:F2})";
        }
    }

    private void DrawBoundingBox(float x, float y, float w, float h)
    {
        // Ekran koordinatlarına çevir
        float screenHeight = Screen.height;
        float screenWidth = Screen.width;

        // UI RawImage'in boyutlarını ve pozisyonunu al
        RectTransform rt = cameraView.GetComponent<RectTransform>();
        Vector2 size = rt.rect.size;
        Vector2 position = rt.position;

        // Koordinatları UI içinde ölçekle
        float scaledX = (x / inputTexture.width) * size.x + position.x - (size.x / 2);
        float scaledY = (1 - (y / inputTexture.height)) * size.y + position.y - (size.y / 2);
        float scaledW = (w / inputTexture.width) * size.x;
        float scaledH = (h / inputTexture.height) * size.y;

        // GUI koordinatlarına çevir
        scaledY = screenHeight - scaledY;

        boundingBoxPositions.Add(new Rect(scaledX, scaledY, scaledW, scaledH));
    }

    private void OnGUI()
    {
        // Tespit kutularını çiz
        GUI.color = Color.red;
        foreach (var box in boundingBoxPositions)
        {
            // Kutunun kenarlarını çiz
            GUI.DrawTexture(new Rect(box.x, box.y, 2, box.height), Texture2D.whiteTexture); // Sol kenar
            GUI.DrawTexture(new Rect(box.x + box.width, box.y, 2, box.height), Texture2D.whiteTexture); // Sağ kenar
            GUI.DrawTexture(new Rect(box.x, box.y, box.width, 2), Texture2D.whiteTexture); // Üst kenar
            GUI.DrawTexture(new Rect(box.x, box.y + box.height, box.width, 2), Texture2D.whiteTexture); // Alt kenar
        }
        boundingBoxPositions.Clear(); // Her frame'de listeyi temizle
    }

    private void UpdateFPS()
    {
        fpsAccumulator += Time.deltaTime;
        fpsFrames++;

        if (fpsAccumulator >= fpsUpdateInterval)
        {
            float fps = fpsFrames / fpsAccumulator;
            frameRateText.text = $"FPS: {fps:F1}";
            fpsAccumulator = 0;
            fpsFrames = 0;
        }
    }

    public class Detection
    {
        public Rect rect;
        public int classId;
        public float score;
    }
}
