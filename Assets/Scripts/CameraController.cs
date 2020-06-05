using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Experimental.XR;
using UnityEngine.UI;
using UnityEngine.XR.ARExtensions;
using UnityEngine.XR.ARFoundation;

/// <summary>
/// This component tests getting the latest camera image
/// and converting it to RGBA format. If successful,
/// it displays the image on the screen as a RawImage
/// and also displays information about the image.
/// 
/// This is useful for computer vision applications where
/// you need to access the raw pixels from camera image
/// on the CPU.
/// 
/// This is different from the ARCameraBackground component, which
/// efficiently displays the camera image on the screen. If you
/// just want to blit the camera texture to the screen, use
/// the ARCameraBackground, or use Graphics.Blit to create
/// a GPU-friendly RenderTexture.
/// 
/// In this example, we get the camera image data on the CPU,
/// convert it to an RGBA format, then display it on the screen
/// as a RawImage texture to demonstrate it is working.
/// This is done as an example; do not use this technique simply
/// to render the camera image on screen.
/// </summary>
public class CameraController : MonoBehaviour
{
    [SerializeField]
    Text m_ImageInfo;

    /// <summary>
    /// The UI Text used to display information about the image on screen.
    /// </summary>
    public Text imageInfo
    {
        get { return m_ImageInfo; }
        set { m_ImageInfo = value; }
    }

    Texture2D m_Texture;
    ARSessionOrigin arOrigin;

    void OnEnable()
    {
        ARSubsystemManager.cameraFrameReceived += OnCameraFrameReceived;

        arOrigin = FindObjectOfType<ARSessionOrigin>();

        InitTF();
        InitIndicator();
    }

    void OnDisable()
    {
        ARSubsystemManager.cameraFrameReceived -= OnCameraFrameReceived;

        CloseTF();
    }

    unsafe void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
    {
        // Attempt to get the latest camera image. If this method succeeds,
        // it acquires a native resource that must be disposed (see below).
        CameraImage image;
        if (!ARSubsystemManager.cameraSubsystem.TryGetLatestImage(out image))
            return;

        // Display some information about the camera image
        //m_ImageInfo.text = string.Format(
        //   "Image info:\n\twidth: {0}\n\theight: {1}\n\tplaneCount: {2}\n\ttimestamp: {3}\n\tformat: {4}\n\tdebug: {5}",
        //    image.width, image.height, image.planeCount, image.timestamp, image.format, debug);

        // Choose an RGBA format.
        // See CameraImage.FormatSupported for a complete list of supported formats.
        var format = TextureFormat.RGBA32;

        if (m_Texture == null || m_Texture.width != image.width || m_Texture.height != image.height)
            m_Texture = new Texture2D(image.width, image.height, format, false);

        // Convert the image to format, flipping the image across the Y axis.
        // We can also get a sub rectangle, but we'll get the full image here.
        var conversionParams = new CameraImageConversionParams(image, format, CameraImageTransformation.None);

        // Texture2D allows us write directly to the raw texture data
        // This allows us to do the conversion in-place without making any copies.
        var rawTextureData = m_Texture.GetRawTextureData<byte>();
        try
        {
            image.Convert(conversionParams, new IntPtr(rawTextureData.GetUnsafePtr()), rawTextureData.Length);
        }
        finally
        {
            // We must dispose of the CameraImage after we're finished
            // with it to avoid leaking native resources.
            image.Dispose();
        }

        // Apply the updated texture data to our texture
        m_Texture.Apply();

        // Run TensorFlow inference on the texture
        RunTF(m_Texture);
    }

    [SerializeField]
    TextAsset model;

    [SerializeField]
    TextAsset labels;

    [SerializeField]
    GameObject indicator;

    Classifier classifier;
    Detector detector;

    private IList outputs;
    private GameObject arrow;
    public GameObject cursor;
    public GameObject textbox;
    private bool repositionFlag;
    private bool drawLineFlag;
    public Material TransparentMat;
    private List<GameObject> allCylinder;
    private List<GameObject> allCups;
    private List<GameObject> allBottle;
    LineRenderer lineRenderer;
    public Color c1 = Color.yellow;
    public Color c2 = Color.red;
    public int lengthOfLineRenderer = 2;
    public void InitTF()
    {
        // MobileNet
        //classifier = new Classifier(model, labels, output: "MobilenetV1/Predictions/Reshape_1");

        // SSD MobileNet
        detector = new Detector(model, labels,
                                input: "image_tensor");

        // Tiny YOLOv2
        //detector = new Detector(model, labels, DetectionModels.YOLO,
        //width: 416,
        //height: 416,
        //mean: 0,
        //std: 255);
    }

    public void InitIndicator()
    {
        //apple = Instantiate(indicator, new Vector3(0, 0, 0), Quaternion.identity);
        //apple.transform.localScale = new Vector3(0.004f, 0.004f, 0.004f);
        //apple.SetActive(false); 
        arrow = Instantiate(indicator, new Vector3(0, 0, 0), Quaternion.identity);
        arrow.transform.localScale = new Vector3(1.1f, 1.1f, 1.1f);
        arrow.SetActive(false);
        repositionFlag = false;
        drawLineFlag = false;
        allCylinder = new List<GameObject>();
        allCups = new List<GameObject>();
        allBottle = new List<GameObject>();

        lineRenderer = gameObject.AddComponent<LineRenderer>();
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.widthMultiplier = 0.2f;
        lineRenderer.positionCount = lengthOfLineRenderer;

        // A simple 2 color gradient with a fixed alpha of 1.0f.
        float alpha = 1.0f;
        Gradient gradient = new Gradient();
        gradient.SetKeys(
            new GradientColorKey[] { new GradientColorKey(c1, 0.0f), new GradientColorKey(c2, 1.0f) },
            new GradientAlphaKey[] { new GradientAlphaKey(alpha, 0.0f), new GradientAlphaKey(alpha, 1.0f) }
        );
        lineRenderer.colorGradient = gradient;


    }

    public void RunTF(Texture2D texture)
    {
        // MobileNet
        //outputs = classifier.Classify(texture, angle: 90, threshold: 0.05f);

        // SSD MobileNet
        outputs = detector.Detect(m_Texture, angle: 90, threshold: 0.6f);

        // Tiny YOLOv2
        //outputs = detector.Detect(m_Texture, angle: 90, threshold: 0.1f);

        if (repositionFlag)
            for (int i = 0; i < outputs.Count; i++)
            {
                var output = outputs[i] as Dictionary<string, object>;
                //debug = output["detectedClass"].ToString() + "  frst ";// + (output["rect"] as Dictionary<string, float>)["x"].ToString();

                if (output["detectedClass"].Equals("laptop"))
                {
                    DrawApple(output["rect"] as Dictionary<string, float>);
                    break;
                }
            }
    }

    public void CloseTF()
    {
        //classifier.Close();
        detector.Close();
    }

    public void Update()
    {
        if (drawLineFlag)
        {
            lineRenderer.SetPosition(0, new Vector3(transform.position.x, transform.position.y-1, transform.position.z));

            foreach (GameObject cylinder in allCups)
            {
                lineRenderer.SetPosition(1, cylinder.transform.position);
            }
            foreach (GameObject cylinder in allBottle)
            {
                //if (cylndr.tag.Equals("cups")){
                lineRenderer.SetPosition(2, cylinder.transform.position);
                for (int i = 2; i < lengthOfLineRenderer; i++)
                {
                    lineRenderer.SetPosition(i, cylinder.transform.position);

                }
            }
        }
        else
        {
            for (int i = 0; i < lengthOfLineRenderer; i++)
            {
                lineRenderer.SetPosition(i, cursor.transform.position);

            }
            //lineRenderer.SetPosition(0, new Vector3(0, 0, 0));
            //lineRenderer.SetPosition(1, new Vector3(0, 0, 0));
            //lineRenderer.SetPosition(2, new Vector3(0, 0, 0));

        }
        // Bit shift the index of the layer (8) to get a bit mask
        int layerMask = 1 << 8;

        // This would cast rays only against colliders in layer 8.
        // But instead we want to collide against everything except layer 8. The ~ operator does this, it inverts a bitmask.
        layerMask = ~layerMask;

        RaycastHit hit;
        // Does the ray intersect any objects excluding the player layer
        if (Physics.Raycast(transform.position, transform.TransformDirection(Vector3.forward), out hit, Mathf.Infinity, layerMask))
        {
            Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.forward) * hit.distance, Color.yellow);
            //debug = "Did Hit";
            cursor.transform.position = hit.point;
        }
        else
        {
            Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.forward) * 1000, Color.white);
        }

    }
    public void OnGUI()
    {
        if (GUI.Button(new Rect(40, 300, 150, 60), "anker new Obj"))
        {
            //var output = outputs[0] as Dictionary<string, object>;
            //debug = output["detectedClass"].ToString() + "  ff: ";//+ (output["rect"] as Dictionary<string, float>)["x"].ToString();
            DrawObj();
        }
        if (GUI.Button(new Rect(40, 200, 150, 60), "delete all Obj"))
        {
            /// = GameObject.FindGameObjectsWithTag("obj_map");

            foreach (GameObject cylinder in allCylinder)
            {
                Destroy(cylinder);
            }
            allCylinder = new List<GameObject>();
        }
        if (GUI.Button(new Rect(40, 1200, 150, 60), "reposition Laptop"))
        {
            repositionFlag = !repositionFlag;
        }
        if (GUI.Button(new Rect(40, 1300, 150, 60), "search closest Bottle"))
        {
            drawLineFlag = !drawLineFlag;
        }
        if (outputs != null)
        {
            // Classification
            //Utils.DrawOutput(outputs, new Vector2(20, 20), Color.red);

            // Object detection
            Utils.DrawOutput(outputs, Screen.width, Screen.height, Color.yellow);
        }
    }
    private void DrawObj()//var output)
    {
        var output = outputs[0] as Dictionary<string, object>;
        //Dictionary<string, float> rect = output["rect"] as Dictionary<string, float>;
        //var xMin = rect["x"];
        //var yMin = 1 - rect["y"];
        //var xMax = rect["x"] + rect["w"];
        //var yMax = 1 - rect["y"] - rect["h"];

        var pos = GetPosition(0, 0);// (xMin + xMax) / 2 * Screen.width, (yMin + yMax) / 2 * Screen.height);

        //debug = output["detectedClass"].ToString();
        //debug = debug + xMin.ToString()+ yMin.ToString()+ xMax.ToString()+ yMax.ToString();

        GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        GameObject cylinderTransparent = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        var cylinderTrRenderer = cylinderTransparent.GetComponent<Renderer>();
        cylinderTrRenderer.material = TransparentMat; // SetColor("_Color", Color.clear);
        //cylinderTrRenderer.material.SetOverrideTag("RenderType", "Transparent");

        cylinderTransparent.transform.position = pos;
        cylinderTransparent.transform.localScale = new Vector3(0.7f, 0.06f, 0.7f);
        cylinder.transform.position = pos;
        cylinder.transform.localScale = new Vector3(1.2f, 0.05f, 1.2f);

        //cylinder.transform.position = new Vector3(-2, 1, 0);
        var cylinderRenderer = cylinder.GetComponent<Renderer>();
        if (output["detectedClass"].Equals("bottle")) { 
            cylinderRenderer.material.SetColor("_Color", Color.blue);
        cylinder.tag = "bottl";
        allBottle.Add(cylinder);
    }
        if (output["detectedClass"].Equals("cup"))
        {
            cylinderRenderer.material.SetColor("_Color", Color.cyan);
            cylinder.tag = "cups";
            allCups.Add(cylinder);
        }
        if (output["detectedClass"].Equals("cell phone"))
            cylinderRenderer.material.SetColor("_Color", Color.magenta);
        if (output["detectedClass"].Equals("apple"))
            cylinderRenderer.material.SetColor("_Color", Color.red);
        if (output["detectedClass"].Equals("laptop"))
            cylinderRenderer.material.SetColor("_Color", Color.green);
        if (output["detectedClass"].Equals("person"))
        {
            cylinderRenderer.material.SetColor("_Color", Color.yellow);
            cylinderTransparent.transform.localScale = new Vector3(4.2f, 0.06f, 4.2f);
            cylinder.transform.localScale = new Vector3(5, 0.05f, 5);
        }

        // add the cube to a list when you instantiate it
        //GameObject cube = Instantiate(cubePrefab, position, rotation);
        allCylinder.Add(cylinder);
        allCylinder.Add(cylinderTransparent);
    }
    private void DrawApple(Dictionary<string, float> rect)
    {
        var xMin = rect["x"];
        var yMin = 1 - rect["y"];
        var xMax = rect["x"] + rect["w"];
        var yMax = 1 - rect["y"] - rect["h"];

        var pos = GetPosition((xMin + xMax) / 2 * Screen.width, (yMin + yMax) / 2 * Screen.height);
        //if(buffertime > 3sek)
        arrow.SetActive(true);
        arrow.transform.position = pos;
        textbox.SetActive(true);
        textbox.transform.position = pos;
        //eObject.CreatePrimitive(PrimitiveType.Cylinder);
        //GameObject cylinderTransparent = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        //var cylinderTrRenderer = cylinderTransparent.GetComponent<Renderer>();
        //cylinderTrRenderer.material = TransparentMat; 

        //cylinderTransparent.transform.position = pos;
        //cylinderTransparent.transform.localScale = new Vector3(0.7f, 0.06f, 0.7f);
        //cylinderTransparent.tag = "obj_map";
        //cylinder.transform.position = pos;
        //cylinder.transform.localScale = new Vector3(1.7f, 0.05f, 1.7f);
        //cylinder.tag = "obj_map";
    }

    private Vector3 GetPositionkk(float x, float y)
    {
        //ARRaycastManager raycastManager = FindObjectOfType<ARRaycastManager>();
        //var screenCenter = Camera.main.ViewportToScreenPoint(new Vector3(0.5f, 0.5f));
        //var hits = new List<ARRaycastHit>();
        //raycastManager.Raycast(screenCenter, hits, TrackableType.Planes);

        var hits = new List<ARRaycastHit>();

        arOrigin.Raycast(new Vector3(x, y, 0), hits, TrackableType.All);

        if (hits.Count > 0)
        {
            var pose = hits[0].pose;
            return pose.position;
        }

        return new Vector3(0, 0, 0);
    }
    private Vector3 GetPosition(float x, float y)
    {

        // Bit shift the index of the layer (8) to get a bit mask
        int layerMask = 1 << 8;

        // This would cast rays only against colliders in layer 8.
        // But instead we want to collide against everything except layer 8. The ~ operator does this, it inverts a bitmask.
        layerMask = ~layerMask;

        RaycastHit hit;
        // Does the ray intersect any objects excluding the player layer
        if (Physics.Raycast(transform.position, transform.TransformDirection(Vector3.forward), out hit, Mathf.Infinity, layerMask))
        {
            //Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.forward) * hit.distance, Color.yellow);
            return hit.point;
        }
        else
        {
           // Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.forward) * 1000, Color.white);
            return new Vector3(0, 0, 0);
        }

        //var hits = new List<ARRaycastHit>();

        //arOrigin.Raycast(new Vector3(x, y, 0), hits, TrackableType.Planes);

        //if (hits.Count > 0)
        //{
        //    var pose = hits[0].pose;

        //    return pose.position;
        //}

        //return new Vector3(1, 1, 0);
    }
}
