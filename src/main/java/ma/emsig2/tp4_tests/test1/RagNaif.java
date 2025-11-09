package ma.emsig2.tp4_tests.test1;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) {
        // === 1. Vérification de la clé API ===
        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("Clé API Gemini non définie ! Utilise : setx GEMINI_KEY \"ta_clé\"");
            return;
        }

        System.out.println("Clé API détectée. Initialisation du RAG...\n");

        // === 2. Chargement du document PDF ===
        Path cheminPDF = Paths.get("src/main/resources/rag.pdf");
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document doc = FileSystemDocumentLoader.loadDocument(cheminPDF, parser);
        System.out.println("Document chargé avec succès.");

        // === 3. Découpage du document ===
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(doc);
        System.out.println("Nombre de segments : " + segments.size());

        // === 4. Génération des embeddings ===
        EmbeddingModel embedder = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embedder.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);
        System.out.println("Embeddings stockés en mémoire.\n");

        // === 5. Configuration du modèle Gemini ===
        ChatLanguageModel gemini = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.7)
                .build();

        // === 6. Configuration du Retriever ===
        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embedder)
                .maxResults(3)
                .minScore(0.5)
                .build();

        // === 7. Création de l'assistant RAG ===
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(gemini)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(retriever)
                .build();

        System.out.println("Assistant RAG prêt !\n");

        // === 8. Boucle interactive ===
        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print("Question > ");
            String question = sc.nextLine().trim();
            if (question.equalsIgnoreCase("exit")) break;

            try {
                String answer = assistant.chat(question);
                System.out.println("\n Réponse : " + answer + "\n");
            } catch (Exception e) {
                System.err.println(" Erreur : " + e.getMessage());
            }
        }
        sc.close();
        System.out.println(" Session terminée.");
    }
}
