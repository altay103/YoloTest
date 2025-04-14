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

    private Interpreter interpreter;
    private float[,,,] inputTensor;
    private float[,,] output0;
    private float[,,] output1;

    private WebCamTexture webCamTexture;
    private Texture2D inputTexture;

    private int inputWidth = 416;
    private int inputHeight = 416;

    private List<Detection> detections = new List<Detection>();
    private float fpsUpdateInterval = 0.5f;
    private float fpsAccumulator = 0f;
    private int fpsFrames = 0;
    private float fpsTimeLeft;

    private bool isProcessing = false;

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

        inputTexture.SetPixels32(webCamTexture.GetPixels32());
        inputTexture.Apply();

        RenderTexture rt = RenderTexture.GetTemporary(inputWidth, inputHeight);
        Graphics.Blit(inputTexture, rt);
        RenderTexture.active = rt;

        Texture2D resized = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);
        resized.ReadPixels(new Rect(0, 0, inputWidth, inputHeight), 0, 0);
        resized.Apply();

        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        Color32[] pixels = resized.GetPixels32();
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

        Destroy(resized);
    }

    private void RunInference()
    {
        interpreter.SetInputTensorData(0, inputTensor);
        interpreter.Invoke();

        var outputInfo0 = interpreter.GetOutputTensorInfo(0);
        var outputInfo1 = interpreter.GetOutputTensorInfo(1);

        output0 = new float[outputInfo0.shape[0], outputInfo0.shape[1], outputInfo0.shape[2]];
        output1 = new float[outputInfo1.shape[0], outputInfo1.shape[1], outputInfo1.shape[2]];

        interpreter.GetOutputTensorData(0, output0);
        interpreter.GetOutputTensorData(1, output1);

        Debug.Log($"First output box: x={output0[0, 0, 0]}, y={output0[0, 0, 1]}, w={output0[0, 0, 2]}, h={output0[0, 0, 3]}");
        Debug.Log($"First output score: class 0 score={output1[0, 0, 0]}");
    }

    private void ProcessOutput()
    {
        detections.Clear();

        int numDetections = output0.GetLength(1);
        int numClasses = output1.GetLength(2);

        for (int i = 0; i < numDetections; i++)
        {
            float x = output0[0, i, 0];
            float y = output0[0, i, 1];
            float w = output0[0, i, 2];
            float h = output0[0, i, 3];

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

            float score = maxProb;
            if (score < scoreThreshold) continue;

            Rect rect = new Rect(x - w / 2, y - h / 2, w, h);
            detections.Add(new Detection
            {
                rect = rect,
                classId = classId,
                score = score
            });
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
            // Kutu koordinatları: x, y, genişlik, yükseklik
            float x = detection.rect.x * inputTexture.width;
            float y = detection.rect.y * inputTexture.height;
            float w = detection.rect.width * inputTexture.width;
            float h = detection.rect.height * inputTexture.height;

            // Kutu çizimi
            DrawBoundingBox(x, y, w, h);

            // Skoru ve sınıfı logla
            Debug.Log($"Detection: Class {detection.classId}, Score: {detection.score}");
        }
    }

    private void DrawBoundingBox(float x, float y, float w, float h)
    {
        Vector3[] worldCorners = new Vector3[4];
        worldCorners[0] = new Vector3(x, y, 0);
        worldCorners[1] = new Vector3(x + w, y, 0);
        worldCorners[2] = new Vector3(x, y + h, 0);
        worldCorners[3] = new Vector3(x + w, y + h, 0);

        // Dünyadaki köşeleri ekran köşelerine dönüştürme
        Vector3[] screenCorners = new Vector3[4];
        for (int i = 0; i < worldCorners.Length; i++)
        {
            screenCorners[i] = Camera.main.WorldToScreenPoint(worldCorners[i]);
        }

        // Kutu çizimi
        Debug.DrawLine(screenCorners[0], screenCorners[1], Color.red);
        Debug.DrawLine(screenCorners[1], screenCorners[3], Color.red);
        Debug.DrawLine(screenCorners[3], screenCorners[2], Color.red);
        Debug.DrawLine(screenCorners[2], screenCorners[0], Color.red);
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
