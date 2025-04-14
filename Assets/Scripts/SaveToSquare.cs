using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SaveToSquare : MonoBehaviour
{
    void Start()
    {
        // Get the RectTransform component
        RectTransform rectTransform = GetComponent<RectTransform>();
        if (rectTransform != null)
        {
            // Set horizontal stretch
            rectTransform.anchorMin = new Vector2(0, rectTransform.anchorMin.y);
            rectTransform.anchorMax = new Vector2(1, rectTransform.anchorMax.y);
            rectTransform.sizeDelta = new Vector2(0, rectTransform.sizeDelta.y);

            // Wait for next frame to get actual width after stretch
            StartCoroutine(SetSquareSize());
        }
    }

    IEnumerator SetSquareSize()
    {
        yield return null;
        RectTransform rectTransform = GetComponent<RectTransform>();
        float width = rectTransform.rect.width;
        rectTransform.sizeDelta = new Vector2(0, width);
    }

    void Update()
    {

    }
}
