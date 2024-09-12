import streamlit as st
import os
import time
from groq import Groq
import io
import chardet
import PyPDF2
import docx

# Supported models
SUPPORTED_MODELS = {
    "Llama 3 70B": "llama3-70b-8192",
    "Llama 3 8B": "llama3-8b-8192",
    "Llama 3.1 70B": "llama-3.1-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 2 9B": "gemma2-9b-it"
}

# Initialize Groq client with API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please set it and restart the app.")
    st.stop()

client = Groq(api_key=groq_api_key)

# Function to extract text from uploaded files
def extract_text_from_file(file_content, file_name):
    if file_name.endswith('.txt'):
        encoding = chardet.detect(file_content)['encoding']
        return file_content.decode(encoding or 'utf-8', errors='replace')
    elif file_name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        return "\n".join(page.extract_text() for page in pdf_reader.pages)
    elif file_name.endswith('.docx'):
        doc = docx.Document(io.BytesIO(file_content))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        return "Unsupported file type"

# Enhanced System Prompt Logic
def enhanced_system_prompt(query):
    prompt = f"""
    <thinking>
    1. Begin analyzing the question: "{query}"
    2. Plan of action:
        a. Briefly outline the approach.
        b. Present a step-by-step reasoning process.
        c. Use "Chain of Thought" reasoning if needed, breaking it into steps.
        d. Consider alternative solutions if applicable.
    </thinking>

    <reflection>
    1. Review reasoning.
    2. Check for potential errors, optimizations, or enhancements.
    3. Reflect on alternative approaches, if any.
    </reflection>

    <output>
    Provide final answer based on the above reasoning and reflection.
    </output>
    """
    return prompt

# Function to handle multi-path reasoning using the ReAct framework
def multi_path_reasoning(selected_task):
    if selected_task == "Research and Information Retrieval":
        return """
        Thought 1: I need to gather detailed information about the impact of climate change on coastal cities. 
        Act 1: Search["impact of climate change on coastal cities"] 
        Obs 1: The search results include scientific articles, government reports, and case studies on the impact of rising sea levels on coastal cities.
        Thought 2: The information seems scattered. I need to focus on retrieving case studies from government reports.
        Act 2: Search["case studies from government reports on rising sea levels"] 
        Obs 2: Found multiple case studies from NOAA and the EPA detailing the impact on specific cities. 
        Thought 3: I have enough case studies but need to summarize key points.
        Act 3: Summarize["key points from case studies on rising sea levels"] 
        Act 4: Finish[summary of key points]
        """
    elif selected_task == "Code Debugging":
        return """
        Thought 1: I need to debug a Python script that’s throwing a TypeError.
        Act 1: Review["Python script TypeError"] 
        Obs 1: The TypeError is due to a mismatch in data types when calling a function.
        Thought 2: I should identify the exact line causing the error and check the data types involved.
        Act 2: Inspect["line of code causing TypeError and data types"] 
        Obs 2: The error is occurring because an integer is being passed where a string is expected.
        Thought 3: I need to correct the data type mismatch and rerun the script.
        Act 3: Modify["correct data type from integer to string"] 
        Act 4: Finish[rerun the script]
        """
    elif selected_task == "Content Generation":
        return """
        Thought 1: I need to write a blog post on the benefits of AI in healthcare.
        Act 1: Generate["outline for blog post on AI in healthcare"] 
        Obs 1: The outline includes sections on diagnostic tools, personalized medicine, and operational efficiency.
        Thought 2: I should expand the section on personalized medicine with examples.
        Act 2: Research["examples of personalized medicine using AI"] 
        Obs 2: Found examples of AI-driven treatments for cancer and diabetes.
        Thought 3: I can now draft the personalized medicine section with these examples.
        Act 3: Write["draft section on personalized medicine with AI examples"] 
        Act 4: Finish[draft complete]
        """
    elif selected_task == "Strategic Planning":
        return """
        Thought 1: I need to develop a strategic plan for increasing customer retention.
        Act 1: Identify["key factors affecting customer retention"] 
        Obs 1: Key factors include product satisfaction, customer support quality, and engagement strategies.
        Thought 2: I should focus on improving customer support and engagement strategies.
        Act 2: Develop["action plan for improving customer support and engagement"] 
        Obs 2: Created a plan including personalized communication, regular feedback loops, and loyalty programs.
        Thought 3: I need to present this plan to the executive team.
        Act 3: Prepare["presentation for executive team on customer retention strategies"] 
        Act 4: Finish[presentation ready]
        """
    return ""

# Function to handle advanced steps: prompt improvement, response, review, and analysis
def advanced_steps(query, model_id):
    try:
        # Step 1: Improve the original prompt
        with st.spinner("Improving the prompt..."):
            improved_prompt_response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert at refining prompts."},
                    {"role": "user", "content": f"Please improve the following prompt for optimal results:\n{query}"}
                ],
                model=model_id,
                max_tokens=500,
            )
            improved_prompt = improved_prompt_response.choices[0].message.content

        # Step 2: Get the response for the improved prompt
        with st.spinner("Generating response for the improved prompt..."):
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": improved_prompt}],
                model=model_id,
                max_tokens=1000,
            )
            generated_response = response.choices[0].message.content

        # Step 3: Review and grade the response
        with st.spinner("Reviewing and grading the response..."):
            review_response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert reviewer and grader."},
                    {"role": "user", "content": f"Please review and grade the following response:\n{generated_response}"}
                ],
                model=model_id,
                max_tokens=500,
            )
            review_feedback = review_response.choices[0].message.content

        # Step 4: Analyze and summarize review points and responses
        with st.spinner("Analyzing and summarizing review points..."):
            analysis_response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert in summarizing and analyzing feedback."},
                    {"role": "user", "content": f"Please analyze and summarize the following review feedback:\n{review_feedback}"}
                ],
                model=model_id,
                max_tokens=500,
            )
            analysis_summary = analysis_response.choices[0].message.content

        return improved_prompt, generated_response, review_feedback, analysis_summary

    except Exception as e:
        st.error(f"An error occurred during advanced steps: {str(e)}")
        return None, None, None, None

# Function to search and summarize using Groq API
def search_and_summarize(query, model_id, system_prompt, context="", reasoning_type="Single-path", selected_task=None):
    try:
        with st.spinner("Searching and summarizing..."):
            if reasoning_type == "Multi-path" and selected_task:
                prompt = multi_path_reasoning(selected_task)
                query = f"{query}\n\n{prompt}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}\n\nReasoning Type: {reasoning_type}\n\nPlease provide a summary and details for this query, considering the given context if relevant."}
            ]
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_id,
                max_tokens=1000,
            )
            response = chat_completion.choices[0].message.content
            summary, details = response.split("\n\n", 1) if "\n\n" in response else (response, "No detailed information available.")
        return summary, details
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None

# Streamlit UI layout
st.set_page_config(layout="wide", page_title="Enhanced Groq Search App")
st.image("p1.png", width=160)

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state.conversations = {"2024-04-15 15:27:16": {"summary": "", "details": "", "system_prompt": "You are a helpful assistant that provides summaries and details based on user queries and given context."}}
if "files" not in st.session_state:
    st.session_state.files = {}
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Mixtral 8x7B"
if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = "2024-04-15 15:27:16"
if "reasoning_type" not in st.session_state:
    st.session_state.reasoning_type = "Single-path"
if "selected_task" not in st.session_state:
    st.session_state.selected_task = "Research and Information Retrieval"

# Sidebar for model selection, conversations, reasoning type, and file management
with st.sidebar:
    st.header("Model Selection")
    st.session_state.selected_model = st.selectbox("Select a model", list(SUPPORTED_MODELS.keys()))

    st.header("Conversations")
    selected_conversation = st.selectbox("Select a conversation", list(st.session_state.conversations.keys()), index=list(st.session_state.conversations.keys()).index(st.session_state.active_conversation))
    
    if st.button("New Conversation"):
        new_conv = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.conversations[new_conv] = {"summary": "", "details": "", "system_prompt": "You are a helpful assistant that provides summaries and details based on user queries and given context."}
        st.session_state.active_conversation = new_conv
        st.rerun()
    
    if st.button("Delete Conversation"):
        if len(st.session_state.conversations) > 1:
            del st.session_state.conversations[selected_conversation]
            st.session_state.active_conversation = list(st.session_state.conversations.keys())[0]
            st.rerun()
        else:
            st.warning("Cannot delete the last conversation.")
    
    new_name = st.text_input("Rename", value=selected_conversation)
    if new_name and new_name != selected_conversation:
        st.session_state.conversations[new_name] = st.session_state.conversations.pop(selected_conversation)
        st.session_state.active_conversation = new_name
        st.rerun()

    st.header("Advanced Prompt Reasoning")
    st.session_state.reasoning_type = st.radio(
        "Select Reasoning Type",
        ("Single-path", "Multi-path", "Advance Steps", "Enhanced System Prompt")
    )
    
    if st.session_state.reasoning_type == "Multi-path":
        st.session_state.selected_task = st.selectbox(
            "Select Task Type",
            ("Research and Information Retrieval", "Code Debugging", "Content Generation", "Strategic Planning")
        )
    
    st.header("File Upload")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        file_contents = uploaded_file.read()
        text_content = extract_text_from_file(file_contents, uploaded_file.name)
        st.session_state.files[uploaded_file.name] = text_content
        st.success(f"File {uploaded_file.name} uploaded and processed successfully!")
    
    st.header("File Index")
    selected_file = st.selectbox("Select a file", list(st.session_state.files.keys()))
    if st.button("Generate Report") and selected_file:
        context = st.session_state.files[selected_file]
        model_id = SUPPORTED_MODELS[st.session_state.selected_model]
        reasoning_type = st.session_state.reasoning_type
        selected_task = st.session_state.selected_task if reasoning_type == "Multi-path" else None
        system_prompt = st.session_state.conversations[st.session_state.active_conversation].get("system_prompt", "")

        if reasoning_type == "Advance Steps":
            with st.spinner("Executing advanced steps..."):
                improved_prompt, generated_response, review_feedback, analysis_summary = advanced_steps(context, model_id)
                st.session_state.conversations[st.session_state.active_conversation]["summary"] = analysis_summary
                st.session_state.conversations[st.session_state.active_conversation]["details"] = f"Improved Prompt:\n{improved_prompt}\n\nGenerated Response:\n{generated_response}\n\nReview Feedback:\n{review_feedback}"
        elif reasoning_type == "Enhanced System Prompt":
            with st.spinner("Generating enhanced prompt response..."):
                system_prompt = enhanced_system_prompt(context)
                response = client.chat.completions.create(
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": context}],
                    model=model_id,
                    max_tokens=1000,
                )
                generated_response = response.choices[0].message.content
                st.session_state.conversations[st.session_state.active_conversation]["summary"] = generated_response

        else:
            with st.spinner("Generating report..."):
                report_summary, report_details = search_and_summarize(f"Generate a detailed report for the file: {selected_file}", model_id, system_prompt, context, reasoning_type, selected_task)
                if report_summary and report_details:
                    st.session_state.conversations[st.session_state.active_conversation]["summary"] = report_summary
                    st.session_state.conversations[st.session_state.active_conversation]["details"] = report_details
                else:
                    st.session_state.conversations[st.session_state.active_conversation]["summary"] = "Failed to generate report."
                    st.session_state.conversations[st.session_state.active_conversation]["details"] = "An error occurred during report generation."

        st.rerun()

# Main panel for displaying the search input and results
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Chat Input")
    user_input = st.text_input("Enter your query here...")
    if st.button("Send"):
        if user_input:
            context = st.session_state.files.get(selected_file, "") if selected_file else ""
            model_id = SUPPORTED_MODELS[st.session_state.selected_model]
            reasoning_type = st.session_state.reasoning_type
            selected_task = st.session_state.selected_task if reasoning_type == "Multi-path" else None
            system_prompt = st.session_state.conversations[st.session_state.active_conversation].get("system_prompt", "")
            
            if reasoning_type == "Advance Steps":
                with st.spinner("Executing advanced steps..."):
                    improved_prompt, generated_response, review_feedback, analysis_summary = advanced_steps(user_input, model_id)
                    st.session_state.conversations[st.session_state.active_conversation]["summary"] = analysis_summary
                    st.session_state.conversations[st.session_state.active_conversation]["details"] = f"Improved Prompt:\n{improved_prompt}\n\nGenerated Response:\n{generated_response}\n\nReview Feedback:\n{review_feedback}"
            elif reasoning_type == "Enhanced System Prompt":
                with st.spinner("Generating enhanced prompt response..."):
                    system_prompt = enhanced_system_prompt(user_input)
                    response = client.chat.completions.create(
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
                        model=model_id,
                        max_tokens=1000,
                    )
                    generated_response = response.choices[0].message.content
                    st.session_state.conversations[st.session_state.active_conversation]["summary"] = generated_response
                    st.session_state.conversations[st.session_state.active_conversation]["details"] = generated_response
            else:
                summary, details = search_and_summarize(user_input, model_id, system_prompt, context, reasoning_type, selected_task)
                if summary and details:
                    st.session_state.conversations[st.session_state.active_conversation]["summary"] = summary
                    st.session_state.conversations[st.session_state.active_conversation]["details"] = details
        else:
            st.warning("Please enter a query to search.")

    if st.button("Regenerate"):
        if st.session_state.conversations[st.session_state.active_conversation]["summary"]:
            context = st.session_state.files.get(selected_file, "") if selected_file else ""
            model_id = SUPPORTED_MODELS[st.session_state.selected_model]
            reasoning_type = st.session_state.reasoning_type
            selected_task = st.session_state.selected_task if reasoning_type == "Multi-path" else None
            system_prompt = st.session_state.conversations[st.session_state.active_conversation].get("system_prompt", "")

            if reasoning_type == "Advance Steps":
                with st.spinner("Executing advanced steps..."):
                    improved_prompt, generated_response, review_feedback, analysis_summary = advanced_steps(user_input, model_id)
                    st.session_state.conversations[st.session_state.active_conversation]["summary"] = analysis_summary
                    st.session_state.conversations[st.session_state.active_conversation]["details"] = f"Improved Prompt:\n{improved_prompt}\n\nGenerated Response:\n{generated_response}\n\nReview Feedback:\n{review_feedback}"
            elif reasoning_type == "Enhanced System Prompt":
                with st.spinner("Generating enhanced prompt response..."):
                    system_prompt = enhanced_system_prompt(user_input)
                    response = client.chat.completions.create(
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
                        model=model_id,
                        max_tokens=1000,
                    )
                    generated_response = response.choices[0].message.content
                    st.session_state.conversations[st.session_state.active_conversation]["summary"] = generated_response
                    st.session_state.conversations[st.session_state.active_conversation]["details"] = generated_response
            else:
                summary, details = search_and_summarize(user_input, model_id, system_prompt, context, reasoning_type, selected_task)
                if summary and details:
                    st.session_state.conversations[st.session_state.active_conversation]["summary"] = summary
                    st.session_state.conversations[st.session_state.active_conversation]["details"] = details

    st.text_area("Response", value=st.session_state.conversations[st.session_state.active_conversation].get("summary", ""), height=200, key="response_area")

    # Display the editable system prompt below the response box
    st.session_state.conversations[st.session_state.active_conversation]["system_prompt"] = st.text_area("System Prompt", value=st.session_state.conversations[st.session_state.active_conversation].get("system_prompt", ""), height=100, key="system_prompt_area")

with col2:
    st.subheader("Information Panel")
    st.text_area("Details", value=st.session_state.conversations[st.session_state.active_conversation].get("details", ""), height=600, key="details_area")

# Footer with additional options
st.markdown("<div style='text-align: center; color: grey;'>Powered by Groq</div>", unsafe_allow_html=True)
st.info("build by dw 9-12-24")