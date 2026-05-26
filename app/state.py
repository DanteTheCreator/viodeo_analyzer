# In-memory single-session state (one video at a time — internal tool)
state: dict = {
    "video_path":             None,       # local filesystem path
    "video_mime":             "video/mp4",
    "gemini_file_uri":        None,       # Gemini File API URI
    "gemini_file_name":       None,       # Gemini File API name
    "comments":               [],
    "brand_logo_path":        None,
    "brand_logo_mime":        None,
    "brand_logo_gemini_name": None,
}
