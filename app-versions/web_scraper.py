import requests
import pdfkit

def fetch_web_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print("Failed to fetch content from the provided URL. Please check the URL and try again.")
            return None
    except Exception as e:
        print("An error occurred while fetching content:", e)
        return None

def convert_to_pdf(content, output_file=r"C:\Users\VIBGYOR\Desktop\LLM-Mini-Resources\output.pdf"):
    try:
        pdfkit.from_string(content, output_file)
        print("PDF generated successfully:", output_file)
    except Exception as e:
        print("An error occurred while generating PDF:", e)

def main():
    url = input("Enter the URL of the web page: ")
    content = fetch_web_content(url)
    if content:
        convert_to_pdf(content)

if __name__ == "__main__":
    main()
