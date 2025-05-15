export default async function handler(req, res) {
    const { question } = req.body;
  
    // Replace this with your deployed backend URL
    const backendUrl = "https://your-rag-api.onrender.com/query ";
  
    const response = await fetch(backendUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  
    const data = await response.json();
    res.status(200).json(data);
  }