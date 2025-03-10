import google.generativeai as genai
import streamlit as st
from fpdf import FPDF
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import plotly.express as px

output_dir = os.path.expanduser("~\Desktop")
 
# Gemini API key
keys_file = st.secrets["keys_file"]
genai.configure(api_key=keys_file)
def extract_code_from_file(uploaded_file):
    return uploaded_file.read().decode("utf-8")
 
def analyze_code_with_gemini_ai(code_text, language, standards=None):
    prompt = f"""
        You are an AI code analyzer. Please review the following {language} code for logical errors, performance issues, and potential inefficiencies. For each issue found, provide the original problematic code snippet, followed immediately by the improved version of the code.
        Below each improved snippet, provide a brief one- or two-line explanation of why the change enhances efficiency, correctness, or readability. Ensure that all improvements adhere to {standards} coding standards, including naming conventions, formatting, and best practices.
        Do not give the complete improved code, just the snippet that needs to be changed.
 
        As a security expert, you will evaluate the provided code to identify vulnerabilities and risks. Look for common attack vectors such as SQL injection, XSS, buffer overflow, and remote code execution. Examine the code for secure coding practices like input validation, output sanitization, authentication, access controls, and error handling. Based on your findings, provide recommendations for improving the code's security posture and mitigating identified risks. Your report should include:
 
        Ensure the following output format:
            Vulnerability Type: <output>
            Description: <output>
            Severity: <output>
            A snippet of affected code:
                ```
                <output>
                ```
            Mitigation walkthrough:
            <output>
       
            Improved code:
                ```
                <output>
                ```
       
        give the sub headings in bold for the response
       
        Focus on clarity and detail to ensure that your analysis is thorough and understandable.
        Code to analyze:
        {code_text}
    """
 
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=prompt)
    response = model.generate_content(prompt)
    return response.text.strip()
 
def analyze_code_with_gemini_ai_v2(code_text, language):
    prompt2 = f"""
    You are an AI code analyzer. Please provide the following scores out of 100 if it is N/A return 0:\
    score of 100 should indicate that it has a very good scoring
    0 should indicate bad scores and 100 should indicate good scores
    - Vulnerability Score
    - Maintainability Score
    - Performance Score
    - Security Score
    - Readability Score
    - Scalability Score
    - Error Handling Score
    - Code Efficiency Score
    - Memory Usage Score
    - API Security Score
    and nothing other than that should be generated
    Code to analyze:
    {code_text}
    """
    model2 = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=prompt2)
    response2 = model2.generate_content(prompt2)
    return response2.text.strip()
 
def parse_ai_response_to_table(response):
    rows = response.strip().split("\n")
    table_data = []
    scores = {}
    for row in rows:
        if ":" in row:
            metric, score = row.split(":", 1)
            metric = metric.strip()
            score = score.strip().replace(",", "")  # Remove commas from score
            try:
                score = int(score)  # Convert to integer
                table_data.append([metric, score])
                scores[metric] = score
            except ValueError:
                print(f"Invalid score format for {metric}: {score}")
                continue
 
    df = pd.DataFrame(table_data, columns=["Metric", "Score"])
 
    # Corrected styling applied only when necessary
    df_styled = df.style.applymap(lambda x: 'color: green' if int(x) >= 4 else 'color: red', subset=['Score']) \
                       .set_table_styles([
                           {'selector': 'thead th', 'props': [('background-color', '#1f4e79'), ('color', 'white')]},
                           {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f4f4f4')]},
                           {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#ffffff')]},
                           {'selector': 'thead', 'props': [('font-size', '14px'), ('font-weight', 'bold')]},
                       ])  # Correct closing brackets here
    return df_styled, scores
 
def plot_score_chart(scores):
    try:
        df = pd.DataFrame(list(scores.items()), columns=["Metric", "Score"])
        fig = px.bar(df, x="Score", y="Metric", orientation='h',
                     color="Score",
                     color_continuous_scale="Blues",
                     title="Code Analysis Score Breakdown")
       
        st.plotly_chart(fig)
        plt.barh(df["Metric"], df["Score"], color=plt.cm.Blues(df["Score"] / max(df["Score"])))
 
        plt.title("Code Analysis Score Breakdown")
        plt.xlabel("Score")
        plt.ylabel("Metric")
        plt.tight_layout()  
        file_path = os.path.join(output_dir, "score_chart.png")
        print(file_path)
        # Save the plot to the specified file path
        plt.savefig(file_path,dpi=300)
        return file_path
    except:
        return None
 
 
# Mapping file extensions to language and standards
language_map = {
    'py': {'language': 'Python', 'standards': 'PEP 8'},
    'java': {'language': 'Java', 'standards': 'Java Code Conventions'},
    'js': {'language': 'JavaScript', 'standards': 'Airbnb JavaScript Style Guide'}
}
 
def generate_pdf_report(analysis_result, analysis_table, score_chart_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
 
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Code Analysis Report", ln=True, align='C')
    pdf.ln(10)
 
    pdf.set_font("Arial", size=12)
    # Split the response into lines
    if analysis_table is not None:
        pdf.ln(10)  # Add some space before the table
        table_width = 100  # 95 (Metric) + 95 (Score)
        table_x = (pdf.w - table_width) / 2  # Calculate the X position to center the table
 
        # Set the starting position for the table
        pdf.set_x(table_x)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(60, 10, "Metric", 1, 0, "L")
        pdf.cell(40, 10, "Score", 1, 1, "L")  # Table Header
 
        # Add data rows to the table
        table_data = analysis_table.data  # Accessing the raw DataFrame
 
        # Add data rows to the table
        pdf.set_font("Arial", '', 12)
        for index, row in table_data.iterrows():
            # Set the starting position for the table
            pdf.set_x(table_x)
            pdf.cell(60, 10, str(row['Metric']), 1, 0, "L")
            pdf.cell(40, 10, str(row['Score']), 1, 1, "L")
 
    # Add Score Chart (from the temporary file)
    if score_chart_path is not None:
        pdf.ln(10)  # Add space before image
        pdf.image(score_chart_path, x=10, w=180)  # Only add once here
        pdf.ln(10)  # Add space below the image to avoid overlap with text
 
    pdf.ln(10)  # Add space before the analysis text
    lines = analysis_result.split("\n")
 
    inside_code_block = False  # Track if we are inside a code block
    code_block_content = []  # Store lines of a code block
 
    # Set the maximum width for the text (allowing for margins)
    max_width = 190  # Adjust this to leave space for margins
    line_height = 6  # Height of each code line in the PDF
 
    code_bg_color = (230, 230, 230)  # Light gray background color for code block
 
    for line in lines:
        if line == "":
            continue  # Skip empty lines
 
        if line.startswith("```"):  # Handle the start and end of code blocks
            if inside_code_block:
                # End of code block - Print the stored block
                pdf.set_fill_color(*code_bg_color)  # Set background fill color for code block
                pdf.set_text_color(0, 0, 0)  # Set text color for code
                pdf.set_font("Courier", size=10)  # Monospace font for code
 
                # For each line in the code block, just print the raw line without handling indentation
                for code_line in code_block_content:
                    wrapped_lines = pdf.multi_cell(max_width, line_height, code_line, align='L', fill=True)
 
                pdf.ln(4)  # Add spacing after code block
                code_block_content = []  
            inside_code_block = not inside_code_block
        elif inside_code_block:
            code_block_content.append(line)  # Add code to the code block content
        else:
            # This is regular text or heading
            # Handle the **bold** text (only content inside ** should be bold)
            bold_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)  # Replace content inside ** with <b> tags
 
            # Now handle the entire line: replace the <b> tags with bold formatting in PDF
            parts = re.split(r'(<b>.*?</b>)', bold_text)  # Split line by bold parts
            bullet_removed = re.sub(r'^\*[\s]*', '', parts[0])  # Remove bullet point symbol at the start of the line
            parts[0] = bullet_removed.strip()
            for part in parts:
                if part.startswith('<b>') and part.endswith('</b>'):
                    # Apply bold formatting to the part inside <b></b>
                    pdf.set_font("Arial", 'B', 12)
                    pdf.multi_cell(max_width, line_height, part[3:-4].strip())  # Remove <b> and </b> tags and print
                else:
                    # Regular text
                    pdf.set_font("Arial", '', 12)
                    pdf.multi_cell(max_width, line_height, part.strip())  # Regular text without bold
 
    # Save PDF to a file
    pdf_output_path = "code_analysis_report_from_response.pdf"
    pdf.output(pdf_output_path)
 
    return pdf_output_path
 
def security_chatbot(question, analysis_result, analysis_scores):
    prompt = f"""
    You are an AI assistant with expertise in coding languages and cybersecurity.
    Your responses should be based only on the analysis results of the uploaded code.
   
    Code Analysis Results:
    {analysis_result}
 
    Code Scores (out of 100):
    {analysis_scores}
 
    Please answer the following question related to the code analysis results.
    Do not answer questions unrelated to the analysis, and politely ask the user to focus on related questions.
 
    User's Question: {question}
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=prompt)
    response = model.generate_content(prompt)
    answer = response.text.strip()
 
    # Find the Q/A section and remove the initial repetitive analysis response
    if "Chatbot" in answer:
        answer = answer.split("Chatbot", 1)[-1].strip()  # Keep everything after "Chatbot" section
 
    # Initialize qa_history if it doesn't exist
    if 'qa' not in st.session_state:
        st.session_state.qa = []
   
    # Append the Q/A pair to the session state history
    st.session_state.qa.append({"question": question, "answer": answer})
 
    return answer
 
 
 
def main():
    st.set_page_config(page_title="Code Security Analyzer", layout="wide")
 
    st.markdown("<h1 style='text-align: center;'>Code Security Analyzer</h1>", unsafe_allow_html=True)
 
    # Sidebar for uploading and analyzing the code
    uploaded_file = st.sidebar.file_uploader("📂 Upload your code file", type=["py", "java", "js"])
 
    # Reset analysis results if a new file is uploaded or file is cleared
    if uploaded_file is None:
        if 'analysis_result' in st.session_state:
            del st.session_state.analysis_result
        if 'analysis_done' in st.session_state:
            del st.session_state.analysis_done
        if 'table_data' in st.session_state:
            del st.session_state.table_data
        if 'scores' in st.session_state:
            del st.session_state.scores
    else:
        file_extension = uploaded_file.name.split('.')[-1]
        code_text = extract_code_from_file(uploaded_file)
 
        if file_extension in language_map:
            language_info = language_map[file_extension]
            language = language_info['language']
            standards = language_info['standards']
 
            st.subheader(f"Uploaded {language} Code:")
            st.code(code_text, language=file_extension)
 
            analyze_button = st.sidebar.button("Analyze Code")
 
            if analyze_button:
                # Analyze code for mistakes and suggestions (first version)
                analysis_result = analyze_code_with_gemini_ai(code_text, language, standards)
                st.session_state.analysis_result = analysis_result
                st.session_state.analysis_done = True  # Flag to track that analysis is done
 
                # Analyze code for scores (second version)
                analysis_result_v2 = analyze_code_with_gemini_ai_v2(code_text, language)
                df_styled, scores = parse_ai_response_to_table(analysis_result_v2)
                st.session_state.table_data = df_styled
 
                # Store the result in session state
                st.session_state.analysis_result = analysis_result
                st.session_state.table_data = df_styled
                st.session_state.scores = scores
 
            # Check if analysis is done, and then show the results
            if 'analysis_done' in st.session_state and st.session_state.analysis_done:
                # Ensure the content persists after clicking download
                st.subheader("Analysis Result")
                st.write(st.session_state.analysis_result)
                st.subheader("Score Breakdown")
                st.write(st.session_state.table_data)
                score_chart_path = plot_score_chart(st.session_state.scores)
 
                # Generate PDF report only if the analysis is done
                pdf_file = generate_pdf_report(st.session_state.analysis_result, st.session_state.table_data, score_chart_path)
 
                st.sidebar.download_button(
                    label="Download Analysis Report",
                    data=open(pdf_file, "rb"),
                    file_name="code_analysis_report.pdf",
                    mime="application/pdf"
                )
 
        else:
            st.sidebar.warning("Unsupported file type.")
 
    # Security Chatbot Interaction - Only triggered if question is asked
    if uploaded_file is not None and 'analysis_done' in st.session_state and st.session_state.analysis_done:
        question = st.sidebar.text_input("🤖 Ask Security Chatbot about the analysis:")
 
        if question:
            # Call the chatbot function to generate the response
            response = security_chatbot(question, st.session_state.analysis_result, st.session_state.scores)
 
        # Display Chatbot History if it exists
        if 'qa' in st.session_state and st.session_state.qa:
            st.subheader("💬 Chatbot History")
            for index, qa in enumerate(st.session_state.qa, 1):
                st.write(f"**Q{index}:** {qa['question']}")
                st.write(f"**A{index}:** {qa['answer']}")
                st.markdown("---")
 
if __name__ == "__main__":
    main()
