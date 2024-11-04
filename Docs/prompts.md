
# Innovative Video Analysis Prompt Examples for OpenSceneSense

Harness the full potential of **OpenSceneSense** with these innovative video analysis prompt examples. These prompts are categorized into **Frame-Based Analysis**, **Detailed Summaries**, and **Brief Summaries** to align with OpenSceneSense's `AnalysisPrompts` structure. Each example includes a detailed description and potential use cases to inspire your projects.

---

## Table of Contents

1. [Frame-Based Analysis Prompts](#frame-based-analysis-prompts)
2. [Detailed Summary Prompts](#detailed-summary-prompts)
3. [Brief Summary Prompts](#brief-summary-prompts)
4. [Leveraging These Prompts with OpenSceneSense](#leveraging-these-prompts-with-openscenesense)
5. [Best Practices](#best-practices)
6. [Conclusion](#conclusion)

---

## 1. Frame-Based Analysis Prompts

Frame-based analysis focuses on extracting detailed information from individual frames within a video. Below are innovative prompts designed to enhance the granularity and depth of frame analysis.

### 1.1. **Object and Action Detection**

```plaintext
"Analyze this frame by identifying all visible objects and their current states. Describe the actions being performed and the relationships between different objects."
```

**Description:**  
This prompt directs the model to meticulously identify objects within a frame and understand their interactions and actions, providing a detailed snapshot of the scene.

**Use Cases:**
- **Security Surveillance:** Detect and describe suspicious activities in real-time.
- **Retail Analytics:** Monitor customer interactions with products on shelves.
- **Autonomous Vehicles:** Identify obstacles and actions within the vehicle’s environment.

---

### 1.2. **Emotion and Expression Recognition**

```plaintext
"Examine this frame to identify the emotions expressed by individuals. Describe facial expressions, body language, and any contextual cues that indicate their emotional states."
```

**Description:**  
Focuses on interpreting the emotional states of individuals within a frame by analyzing visual cues such as facial expressions and body language.

**Use Cases:**
- **Market Research:** Gauge consumer reactions to products or advertisements.
- **Mental Health:** Analyze patient expressions in therapeutic settings.
- **Entertainment:** Assess character emotions in film and television.

---

### 1.3. **Environmental and Contextual Analysis**

```plaintext
"Assess the environmental elements present in this frame, including lighting, weather conditions, and background settings. Explain how these factors contribute to the overall atmosphere of the scene."
```

**Description:**  
Encourages the model to evaluate the broader environmental context of a frame, understanding how elements like lighting and weather influence the scene's mood.

**Use Cases:**
- **Film Production:** Evaluate shooting conditions and their impact on scene aesthetics.
- **Virtual Reality:** Enhance realistic environment rendering based on contextual analysis.
- **Environmental Monitoring:** Detect and report on environmental changes in real-time footage.

---

### 1.4. **Action Sequence Identification**

```plaintext
"Identify and describe the sequence of actions taking place in this frame. Highlight any significant movements or interactions that are pivotal to the ongoing narrative."
```

**Description:**  
Aims to dissect the flow of actions within a frame, pinpointing key movements that drive the story forward.

**Use Cases:**
- **Sports Analytics:** Analyze player movements and strategies during games.
- **Instructional Videos:** Break down complex tasks into actionable steps.
- **Animation Studios:** Ensure continuity and accuracy in character movements.

---

### 1.5. **Color and Composition Evaluation**

```plaintext
"Evaluate the color palette and composition of this frame. Discuss how the use of color and spatial arrangement enhances the visual appeal and storytelling of the video."
```

**Description:**  
Focuses on the artistic aspects of a frame, assessing how color and composition contribute to the overall quality and narrative.

**Use Cases:**
- **Art Critique:** Provide feedback on visual elements in creative projects.
- **Advertising:** Optimize color schemes for better audience engagement.
- **Content Creation:** Enhance video aesthetics through informed compositional choices.

---

## 2. Detailed Summary Prompts

Detailed summaries provide an in-depth overview of the video's content, integrating both visual and audio elements. These prompts are designed to generate comprehensive narratives that capture the essence of the video.

### 2.1. **Comprehensive Narrative Integration**

```plaintext
"Create a detailed narrative that integrates both visual and audio elements of this video. Include key events, character interactions, and contextual information to provide a full understanding of the content. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}"
```

**Description:**  
This prompt encourages the generation of a thorough and cohesive storyline by combining visual actions with audio transcripts, ensuring a complete depiction of the video.

**Use Cases:**
- **Content Creation:** Develop scripts or storyboards from existing footage.
- **Educational Platforms:** Provide detailed summaries of instructional videos for better learner comprehension.
- **Film and Media Studies:** Analyze and document movie plots for reviews or academic purposes.

---

### 2.2. **Thematic Analysis and Insights**

```plaintext
"Analyze the underlying themes and messages conveyed in this video. Discuss how visual elements and audio contribute to these themes, providing examples from specific moments in the video. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}"
```

**Description:**  
Focuses on identifying and explaining the core themes of the video, illustrating how different elements work together to communicate deeper messages.

**Use Cases:**
- **Literary Analysis:** Examine themes in narrative videos or films.
- **Marketing:** Understand and leverage thematic elements for targeted advertising.
- **Social Research:** Analyze media content to extract societal messages and trends.

---

### 2.3. **Character Development and Interaction**

```plaintext
"Provide a detailed summary of character development and interactions throughout this video. Highlight key moments that showcase character growth, relationships, and dynamics. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}"
```

**Description:**  
Aims to capture the evolution of characters and their relationships, offering insights into their development and interactions within the video.

**Use Cases:**
- **Screenwriting:** Assess character arcs and interactions for storytelling purposes.
- **Psychology:** Study character behaviors and relationships for research.
- **Interactive Media:** Enhance character-driven narratives in games and simulations.

---

### 2.4. **Contextual Event Sequencing**

```plaintext
"Generate a detailed summary outlining the sequence of significant events in this video. Explain how each event leads to the next, providing context and connections between different parts of the video. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}"
```

**Description:**  
Encourages the model to map out the progression of events, ensuring a logical flow and contextual connections throughout the video's timeline.

**Use Cases:**
- **Project Management:** Document project milestones and progress from meeting recordings.
- **Historical Analysis:** Summarize events from documentary footage.
- **Event Planning:** Outline sequences from event recordings for future planning.

---

### 2.5. **Multimodal Contextual Synthesis**

```plaintext
"Combine visual and audio data to synthesize a detailed summary that captures the full context of this video. Include descriptions of scenes, dialogues, and their interrelations to present a unified overview. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}"
```

**Description:**  
Focuses on merging both modalities—visual and audio—to create a unified and comprehensive summary that reflects the complete context of the video.

**Use Cases:**
- **Content Archiving:** Create detailed records of multimedia content for archives.
- **Accessibility Tools:** Provide comprehensive descriptions for visually impaired users.
- **Media Production:** Summarize content for stakeholders and team members.

---

## 3. Brief Summary Prompts

Brief summaries offer concise overviews of the video's content, highlighting the main points without extensive detail. These prompts are ideal for quick insights and easy-to-digest information.

### 3.1. **Concise Content Overview**

```plaintext
"Provide a concise summary that highlights the main visual and audio elements of this video. Focus on key points and essential information to deliver a clear and brief understanding. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"
```

**Description:**  
Generates a short and clear summary that captures the essential aspects of the video, making it easy to grasp the primary content quickly.

**Use Cases:**
- **Social Media:** Create quick video descriptions for platforms like Twitter or Instagram.
- **Content Previews:** Offer brief overviews for video libraries or streaming services.
- **Executive Summaries:** Provide high-level insights for stakeholders and decision-makers.

---

### 3.2. **Highlight Reel Summary**

```plaintext
"Summarize the key highlights of this video, focusing on the most impactful moments and main messages. Ensure the summary is easy to read and provides the complete context in a succinct manner. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"
```

**Description:**  
Emphasizes the most significant and impactful parts of the video, delivering a summary that highlights the core messages and memorable moments.

**Use Cases:**
- **News Outlets:** Create brief summaries of news clips for quick consumption.
- **Marketing:** Highlight key product features or campaign messages.
- **Event Recaps:** Offer concise overviews of events for attendee follow-ups.

---

### 3.3. **Quick Insights Summary**

```plaintext
"Generate a quick summary that captures the primary insights and takeaways from this video. Focus on delivering information that provides immediate understanding without delving into extensive details. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"
```

**Description:**  
Aims to present the main insights and lessons from the video, ensuring that viewers can quickly grasp the fundamental points.

**Use Cases:**
- **Educational Content:** Provide students with quick takeaways from lecture videos.
- **Business Analytics:** Summarize meeting recordings for actionable insights.
- **Content Curation:** Offer brief insights for curated content lists or newsletters.

---

### 3.4. **Executive Brief Summary**

```plaintext
"Create an executive-level brief summary of this video, highlighting strategic points and critical information. Ensure the summary is succinct and tailored for decision-makers who need a high-level overview. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"
```

**Description:**  
Tailored for executives and decision-makers, this prompt focuses on delivering strategic and critical information in a brief format.

**Use Cases:**
- **Business Reports:** Summarize corporate presentations or quarterly reviews.
- **Strategic Planning:** Provide overviews of strategy sessions or planning meetings.
- **Board Meetings:** Offer concise summaries of board discussions and decisions.

---

### 3.5. **Snapshot Summary**

```plaintext
"Provide a snapshot summary of this video, capturing the essence and main points in a few sentences. Ensure the summary is clear, direct, and easily understandable at a glance. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"
```

**Description:**  
Delivers a quick and clear snapshot of the video's content, ideal for situations where brevity and clarity are paramount.

**Use Cases:**
- **Content Indexing:** Create brief entries for video catalogs or databases.
- **Email Newsletters:** Include quick video summaries for subscriber updates.
- - **Mobile Viewing:** Provide concise summaries optimized for quick reading on mobile devices.

---

## 4. Leveraging These Prompts with OpenSceneSense

Integrate these innovative prompts into your **OpenSceneSense** workflows to unlock deeper insights and build more sophisticated video analysis applications. Customize and combine these prompts to suit your specific project needs, whether you're developing interactive applications, enhancing content creation, or conducting advanced research.

### **Getting Started**

1. **Choose Appropriate Prompts:** Select prompts from the categories that best align with your analysis goals—Frame-Based Analysis, Detailed Summaries, or Brief Summaries.
2. **Customize if Needed:** Modify the prompts to better fit your specific use case or to include additional context.
3. **Integrate with OpenSceneSense:** Utilize the prompts within your `AnalysisPrompts` configuration.
4. **Analyze and Iterate:** Run the analysis, review the results, and refine your prompts for optimal outcomes.

### **Example Integration**

Here's how you can integrate these prompts into your **OpenSceneSense** setup:

```python
from openscenesense import ModelConfig, AnalysisPrompts, VideoAnalyzer

# Define custom prompts
custom_prompts = AnalysisPrompts(
    frame_analysis="""
    "Analyze this frame by identifying all visible objects and their current states. Describe the actions being performed and the relationships between different objects."
    """,
    detailed_summary="""
    "Create a detailed narrative that integrates both visual and audio elements of this video. Include key events, character interactions, and contextual information to provide a full understanding of the content. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}"
    """,
    brief_summary="""
    "Provide a concise summary that highlights the main visual and audio elements of this video. Focus on key points and essential information to deliver a clear and brief understanding. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"
    """
)

# Initialize the video analyzer
analyzer = VideoAnalyzer(
    api_key="your-openai-api-key",
    model_config=ModelConfig(
        vision_model="gpt-4o",           # Vision-capable model
        text_model="gpt-4o-mini",        # Chat completion model
        audio_model="whisper-1"          # Whisper model for audio transcription
    ),
    prompts=custom_prompts,
    min_frames=8,
    max_frames=32,
    frame_selector=DynamicFrameSelector(),
    frames_per_minute=8.0,
    log_level=logging.INFO
)

# Analyze the video
video_path = "path/to/your/video.mp4"
results = analyzer.analyze_video(video_path)

# Print the results
print("\nBrief Summary:")
print(results['brief_summary'])

print("\nDetailed Summary:")
print(results['summary'])

print("\nVideo Timeline:")
print(results['timeline'])

print("\nMetadata:")
for key, value in results['metadata'].items():
    print(f"{key}: {value}")
```

---

## 5. Best Practices

To maximize the effectiveness of your prompts within **OpenSceneSense**, consider the following best practices:

### **5.1. Be Specific and Clear**

- **Clarity:** Ensure your prompts are unambiguous and clearly state the analysis objectives.
- **Focus:** Direct the model’s attention to specific elements or aspects of the video to obtain targeted insights.

### **5.2. Use Descriptive Language**

- **Detail:** Incorporate descriptive terms to enhance the model’s understanding and analysis depth.
- **Contextual Cues:** Provide context within prompts to guide the model in generating relevant and accurate outputs.

### **5.3. Integrate Tags Effectively**

- **Utilize `{timeline}`, `{duration}`, and `{transcript}`:** These tags provide essential context, enabling the model to reference specific parts of the video and audio content.
- **Consistency:** Maintain consistent usage of tags across prompts to ensure cohesive and contextually aware analyses.

### **5.4. Iterate and Refine**

- **Testing:** Continuously test and refine your prompts based on the analysis results to improve accuracy and relevance.
- **Feedback Loop:** Use the outputs to inform prompt adjustments, enhancing the quality of future analyses.

### **5.5. Combine Prompts for Comprehensive Insights**

- **Multiple Angles:** Use a combination of frame-based and summary prompts to gather multi-faceted insights.
- **Layered Analysis:** Start with frame analysis to extract detailed information, then use summary prompts to compile and contextualize the findings.

---

## 6. Conclusion

These innovative prompt examples are tailored to fit **OpenSceneSense**'s `AnalysisPrompts` structure, providing specialized prompts for **Frame-Based Analysis**, **Detailed Summaries**, and **Brief Summaries**. By integrating these prompts into your workflows, you can unlock deeper insights and build more sophisticated video analysis applications. Customize and expand upon these examples to explore new possibilities and enhance your projects with intelligent video-centric solutions.


