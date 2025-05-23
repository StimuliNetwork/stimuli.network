const http = require('http');
const fs =require('fs');
const path = require('path');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require("@google/generative-ai");

const PORT = process.env.PORT || 8080;
const GEMINI_API_KEY = process.env.GEMINI_STIMULI_KEY;
const AI_MODEL_NAME = "gemini-2.0-flash";
const COMMENT_API_ENDPOINT = "/api/generate-comment";
const POST_CONTENT_API_ENDPOINT = "/api/generate-post-content";
const ELABORATE_POST_API_ENDPOINT = "/api/elaborate-post";
const REPLY_API_ENDPOINT = "/api/generate-reply";
const PING_ENDPOINT = "/api/ping";

if (!GEMINI_API_KEY) {
    console.error("FATAL ERROR: GEMINI_STIMULI_KEY environment variable is not set. Server cannot start.");
    process.exit(1);
}

let genAI, aiModel;
try {
    genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    aiModel = genAI.getGenerativeModel({ model: AI_MODEL_NAME });
} catch (error) {
    console.error("FATAL ERROR: AI client initialization failed. Server cannot start.", error);
    process.exit(1);
}

const generationSafetySettings = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT,         threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,  threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,  threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
];

function cleanAiTextForParagraphs(text) {
    if (!text) return "";
    // Step 1: Replace literal "\n" (escaped newline) with actual newline character.
    // This handles cases where AI outputs "\\n".
    let cleanedText = text.replace(/\\n/g, '\n');
    // Step 2: Normalize multiple consecutive actual newlines into a single actual newline.
    // This ensures \n\n becomes \n, \n\n\n becomes \n, etc.
    // The client will then split on this single \n.
    cleanedText = cleanedText.replace(/\n+/g, '\n');
    return cleanedText.trim();
}


async function generateSinglePostContent(themeText) {
    if (!themeText || typeof themeText !== 'string' || themeText.trim() === "") {
        return { error: "Invalid theme provided for post content generation" };
    }
    const prompt = `Generate a community update post of about 250-350 characters, consisting of 2-3 paragraphs, expanding on the theme: "${themeText}".
Focus on constructive engagement, community building, or upcoming initiatives.
The output should be plain text. Separate paragraphs with a single newline character (\\n). Do NOT use double newlines (\\n\\n) or any other escape sequences for newlines.
Do not include a title or any preambles like "Here's a post:". Just the post content.`;

    try {
        const result = await aiModel.generateContent({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { temperature: 0.7, topP: 0.95, maxOutputTokens: 512 },
            safetySettings: generationSafetySettings
        });
        const response = result.response;
        if (response) {
            const candidate = response.candidates?.[0];
            if (candidate?.safetyRatings?.some(rating => rating.probability !== 'NEGLIGIBLE' && rating.probability !== 'LOW')) {
                 return { error: "AI response blocked by safety filter" };
            }
            const blockReason = response.promptFeedback?.blockReason;
            if (blockReason) return { error: `AI response blocked: ${blockReason}` };
            if (candidate?.content?.parts?.[0]?.text) {
                // Apply cleaning
                return { postText: cleanAiTextForParagraphs(candidate.content.parts[0].text) };
            }
            return { error: "AI response format unexpected (no text part)." };
        }
        return { error: "AI generation failed: No response received." };
    } catch (error) {
        return { error: `AI generation failed: ${error.message?.substring(0, 100) || "Unknown error"}` };
    }
}

async function generateElaborationContent(themeText, originalPostContext) {
    if (!themeText || typeof themeText !== 'string' || themeText.trim() === "") {
        return { error: "Invalid theme provided for elaboration" };
    }
    const prompt = `A community post was made with the theme: "${themeText}".
The post started with: "${originalPostContext.substring(0, 100)}...".
Please provide a detailed explanation or elaboration (2-3 substantial paragraphs, around 400-600 characters total) on this theme to help someone understand it better.
Focus on clarifying concepts, providing context, or offering different perspectives related to the theme.
The output should be plain text. Separate paragraphs with a single newline character (\\n). Do NOT use double newlines (\\n\\n) or any other escape sequences for newlines.
Do not include a title or any preambles like "Here's an elaboration:". Just the elaboration content.`;

    try {
        const result = await aiModel.generateContent({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { temperature: 0.6, topP: 0.95, maxOutputTokens: 1024 },
            safetySettings: generationSafetySettings
        });
        const response = result.response;
        if (response) {
            const candidate = response.candidates?.[0];
            if (candidate?.safetyRatings?.some(rating => rating.probability !== 'NEGLIGIBLE' && rating.probability !== 'LOW')) {
                 return { error: "AI response blocked by safety filter" };
            }
            const blockReason = response.promptFeedback?.blockReason;
            if (blockReason) return { error: `AI response blocked: ${blockReason}` };
            if (candidate?.content?.parts?.[0]?.text) {
                // Apply cleaning
                return { elaborationText: cleanAiTextForParagraphs(candidate.content.parts[0].text) };
            }
            return { error: "AI response format unexpected (no text part)." };
        }
        return { error: "AI generation failed: No response received." };
    } catch (error) {
        return { error: `AI generation failed: ${error.message?.substring(0, 100) || "Unknown error"}` };
    }
}

async function generateMultipleComments(postContext) {
    if (!postContext || typeof postContext !== 'string' || postContext.trim() === "") {
        return { error: "Invalid context provided" };
    }
    const minComments = 10;
    const maxComments = 25;
    const numberOfComments = Math.floor(Math.random() * (maxComments - minComments + 1)) + minComments;

    const prompt = `Based on the following online post snippet: "${postContext}"
Generate exactly ${numberOfComments} **highly distinct and varied** comments reacting to the post. Ensure each comment offers a **unique perspective or angle** compared to the others. Comments should be short (10-25 words each), realistic, relevant, constructive, and creative.
Comments should aim to be **thought-provoking**, supportive, curious, **offer an insightful perspective,** or provide a brief related thought that **builds upon the post's idea**.
**Crucially, avoid repeating similar phrases or sentence structures across the comments.**
Do not use hashtags. Do not introduce yourself (e.g., "As an AI..."). Avoid generic questions unless they genuinely add significant value or insight.
Output ONLY a valid JSON array containing exactly ${numberOfComments} strings, where each string is one comment. Example format: ["Comment 1 text.", "Comment 2 text.", ..., "Comment ${numberOfComments} text."]`;

    try {
        const result = await aiModel.generateContent({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { temperature: 0.9, topP: 0.95, maxOutputTokens: 2048 },
            safetySettings: generationSafetySettings
        });
        const response = result.response;
        if (response) {
            const candidate = response.candidates?.[0];
            if (candidate?.safetyRatings?.some(rating => rating.probability !== 'NEGLIGIBLE' && rating.probability !== 'LOW')) {
                 return { error: "AI response blocked by safety filter" };
            }
            const blockReason = response.promptFeedback?.blockReason;
            if (blockReason) return { error: `AI response blocked: ${blockReason}` };

            if (candidate?.content?.parts?.[0]?.text) {
                const rawText = candidate.content.parts[0].text.trim();
                let jsonString = rawText;
                const jsonRegex = /^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$/;
                const match = rawText.match(jsonRegex);
                if (match && match[1]) jsonString = match[1].trim();
                try {
                    let parsedComments = JSON.parse(jsonString);
                    if (Array.isArray(parsedComments) && parsedComments.every(item => typeof item === 'string')) {
                        return { comments: parsedComments };
                    }
                    return { error: "AI response format incorrect (not an array of strings)." };
                } catch (parseError) {
                    return { error: "AI response format incorrect (failed to parse JSON)." };
                }
            }
            return { error: "AI response format unexpected (no text part)." };
        }
        return { error: "AI generation failed: No response received." };
    } catch (error) {
        return { error: `AI generation failed: ${error.message?.substring(0, 100) || "Unknown error"}` };
    }
}

async function generateSingleReply(parentCommentText) {
    if (!parentCommentText || typeof parentCommentText !== 'string' || parentCommentText.trim() === "") {
        return { error: "Invalid parent comment text provided for reply generation" };
    }

    const prompt = `Given the following comment from an online discussion:
"${parentCommentText}"
Generate a short, relevant, and engaging reply to this comment (around 5-15 words).
The reply should be conversational and constructive.
Do not introduce yourself (e.g., "As an AI...").
Output ONLY the reply text as a single string. Do not use JSON, arrays, or any other formatting. Just the plain text of the reply.`;

    try {
        const result = await aiModel.generateContent({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { temperature: 0.75, topP: 0.95, maxOutputTokens: 64 },
            safetySettings: generationSafetySettings
        });
        const response = result.response;
        if (response) {
            const candidate = response.candidates?.[0];
            if (candidate?.safetyRatings?.some(rating => rating.probability !== 'NEGLIGIBLE' && rating.probability !== 'LOW')) {
                 return { error: "AI reply blocked by safety filter" };
            }
            const blockReason = response.promptFeedback?.blockReason;
            if (blockReason) return { error: `AI reply blocked: ${blockReason}` };
            if (candidate?.content?.parts?.[0]?.text) {
                const replyText = candidate.content.parts[0].text.trim();
                const cleanedReplyText = replyText.replace(/^```(?:text)?\s*([\s\S]*?)\s*```$/s, '$1').trim();
                return { replyText: cleanedReplyText };
            }
            return { error: "AI reply format unexpected (no text part)." };
        }
        return { error: "AI reply generation failed: No response received." };
    } catch (error) {
        return { error: `AI reply generation failed: ${error.message?.substring(0, 100) || "Unknown error"}` };
    }
}

const server = http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS, GET');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
    }

    const handleRequest = async (generationFunction, requestDataExtractor, successKey, errorContext) => {
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('error', (err) => {
            if (!res.headersSent) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: "Request body error" }));
            }
        });
        req.on('end', async () => {
            try {
                if (!body) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    return res.end(JSON.stringify({ error: "Request body cannot be empty." }));
                }
                let requestData;
                try {
                    requestData = JSON.parse(body);
                } catch (parseError) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    return res.end(JSON.stringify({ error: "Invalid JSON format in request body." }));
                }

                const extractedData = requestDataExtractor(requestData);
                if (extractedData.error) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    return res.end(JSON.stringify({ error: extractedData.error }));
                }

                const generationResult = await generationFunction(...extractedData.args);

                if (generationResult.error) {
                    const statusCode = (generationResult.error.includes("blocked") || generationResult.error.includes("format incorrect") || generationResult.error.includes("Invalid")) ? 400 : 500;
                    res.writeHead(statusCode, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: `Failed to generate ${errorContext}: ${generationResult.error}` }));
                } else {
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ [successKey]: generationResult[successKey] }));
                }
            } catch (error) {
                if (!res.headersSent) {
                    res.writeHead(500, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: `Internal server error processing ${errorContext}.` }));
                }
            }
        });
    };

    if (req.method === 'POST' && req.url === COMMENT_API_ENDPOINT) {
        handleRequest(
            generateMultipleComments,
            (data) => {
                const context = data.context;
                if (!context || typeof context !== 'string') return { error: "Request body must contain a 'context' field as a string." };
                return { args: [context] };
            },
            'comments',
            'comments'
        );
    } else if (req.method === 'POST' && req.url === POST_CONTENT_API_ENDPOINT) {
        handleRequest(
            generateSinglePostContent,
            (data) => {
                const theme = data.theme;
                if (!theme || typeof theme !== 'string') return { error: "Request body must contain a 'theme' field as a string." };
                return { args: [theme] };
            },
            'postText',
            'post content'
        );
    } else if (req.method === 'POST' && req.url === ELABORATE_POST_API_ENDPOINT) {
        handleRequest(
            generateElaborationContent,
            (data) => {
                const theme = data.theme;
                const originalPostContext = data.originalPostContext || "";
                if (!theme || typeof theme !== 'string') return { error: "Request body must contain a 'theme' field as a string." };
                return { args: [theme, originalPostContext] };
            },
            'elaborationText',
            'elaboration'
        );
    } else if (req.method === 'POST' && req.url === REPLY_API_ENDPOINT) {
        handleRequest(
            generateSingleReply,
            (data) => {
                const parentCommentText = data.parentCommentText;
                if (!parentCommentText || typeof parentCommentText !== 'string') return { error: "Request body must contain a 'parentCommentText' field as a string." };
                return { args: [parentCommentText] };
            },
            'replyText',
            'reply'
        );
    } else if (req.method === 'GET' && req.url === PING_ENDPOINT) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'awake' }));
    } else if (req.method === 'GET' && req.url === '/') {
        const filePath = path.join(__dirname, 'index.html');
        fs.readFile(filePath, 'utf8', (err, content) => {
            if (err) {
                if (err.code === 'ENOENT') {
                    res.writeHead(404, { 'Content-Type': 'text/plain' });
                    res.end('404 Not Found: index.html missing.');
                } else {
                    res.writeHead(500, { 'Content-Type': 'text/plain' });
                    res.end('500 Internal Server Error.');
                }
            } else {
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.end(content);
            }
        });
    } else {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Not Found');
    }
});

server.listen(PORT, () => { });

server.on('error', (error) => {
    console.error("Server failed to start:", error);
    if (error.code === 'EADDRINUSE') {
        console.error(`Port ${PORT} is already in use. Is another server running?`);
    }
    process.exit(1);
});

function shutdown() {
  server.close(() => {
    process.exit(0);
  });
  setTimeout(() => {
    console.error('Could not close connections in time, forcefully shutting down');
    process.exit(1);
  }, 10000);
}
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
