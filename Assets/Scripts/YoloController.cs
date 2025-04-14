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

    private void Start()
    {
        try
        {
            string modelPath = System.IO.Path.Combine(Application.streamingAssetsPath, modelFile);
            Debug.Log($"Model path: {modelPath}");
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
        StartCoroutine(ProcessFrame());
    }

    private IEnumerator ProcessFrame()
    {
        yield return new WaitForEndOfFrame();

        ProcessInputTexture();
        RunInference();
        ProcessOutput();
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
        // Buraya çizim veya UI göstergeleri eklenebilir
    }

    private void UpdateFPS()
    {
        fpsTimeLeft -= Time.deltaTime;
        fpsAccumulator += Time.timeScale / Time.deltaTime;
        fpsFrames++;

        if (fpsTimeLeft <= 0f)
        {
            float fps = fpsAccumulator / fpsFrames;
            frameRateText.text = $"FPS: {fps:0.}";
            fpsTimeLeft = fpsUpdateInterval;
            fpsAccumulator = 0f;
            fpsFrames = 0;
        }
    }

    private void OnDestroy()
    {
        if (interpreter != null)
        {
            interpreter.Dispose();
        }

        if (webCamTexture != null)
        {
            webCamTexture.Stop();
        }

        if (inputTexture != null)
        {
            Destroy(inputTexture);
        }
    }

    private struct Detection
    {
        public Rect rect;
        public int classId;
        public float score;
    }
}
