<div align="center">

<img src="landing_page/styles/images/logo.png" alt="Resumai Logo" width="600"/>

</div>

<h1 align="center">
QueryQuack
</h1>

<h4 align="center">
Quack the query, crack the PDF!
</h1>


<p align="center">
    <a href="https://github.com/negativenagesh/Resumai/stargazers"><img src="https://img.shields.io/github/stars/negativenagesh/QueryQuack" alt="Stars Badge"/></a>
    <a href="https://github.com/negativenagesh/Resumai/network/members"><img src="https://img.shields.io/github/forks/negativenagesh/QueryQuack" alt="Forks Badge"/></a>
    <a href="https://github.com/negativenagesh/Resumai/pulls"><img src="https://img.shields.io/github/issues-pr/negativenagesh/QueryQuack" alt="Pull Requests Badge"/></a>
    <a href="https://github.com/negativenagesh/Resumai/issues"><img src="https://img.shields.io/github/issues/negativenagesh/QueryQuack" alt="Issues Badge"/></a>
    <a href="https://github.com/negativenagesh/Resumai/blob/main/LICENSE"><img src="https://img.shields.io/github/license/negativenagesh/QueryQuack" alt="License Badge"/></a>
</p>

<div style= "padding: 20px; border-radius: 10px; animation: bounceIn 2s;"> <h2 style="color: #00d4ff;">Setup Instructions</h2> <p style="color: #b0b0b3;"> Ready to launch your career rocket? Follow these steps to set up Resumai locally! 🛠️ </p>

1. Clone the Repository
```bash
git clone https://github.com/negativenagesh/QueryQuack.git
cd Resumai
```

2. Create a venv
```python
python3 -m venv .venv
source .venv/bin/activate
```  

3. Install Dependencies
```python
pip install -r pkgs.txt
```

4. Set up Gemini API
```bash
# Create a .env file in the project directory
touch .env

# Add your Gemini API key to the .env file
echo "api_key=your_api_key_here" >> .env
```