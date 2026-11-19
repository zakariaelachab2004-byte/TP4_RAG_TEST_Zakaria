package ma.emsi.elachab.tp4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TestRoutage {

    public static void main(String[] args) {

        String apiKey = System.getenv("GEMINI_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            System.out.println("Erreur : GEMINI_KEY non définie.");
            return;
        }

        // 1. Configuration du modèle Gemini
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.2)
                .logRequests(true)
                .logResponses(true)
                .build();

        // 2. Modèle d'embeddings partagé
        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        // 3. Base documentaire n°1 : cours RAG
        ContentRetriever retrieverRag = creerRetriever(
                "rag.pdf",
                new ApacheTikaDocumentParser(),
                embeddingModel
        );

        // 4. Base documentaire n°2 : cours Java
        ContentRetriever retrieverJava = creerRetriever(
                "java.txt",
                new TextDocumentParser(),
                embeddingModel
        );

        // 5. Descriptions pour le routage
        Map<ContentRetriever, String> descriptions =
                new LinkedHashMap<>();

        descriptions.put(
                retrieverRag,
                "Support de cours sur le RAG, le fine-tuning, " +
                        "les embeddings, les bases vectorielles " +
                        "et LangChain4j."
        );

        descriptions.put(
                retrieverJava,
                "Support de cours sur la programmation Java, " +
                        "l'héritage, le polymorphisme, les collections, " +
                        "les Streams, les threads et JavaFX."
        );

        // 6. Routeur intelligent utilisant Gemini
        QueryRouter routeurGemini =
                new LanguageModelQueryRouter(model, descriptions);

        // 7. Afficher explicitement la source choisie
        QueryRouter routeurAvecLogs = query -> {

            System.out.println(
                    "\n[ROUTAGE] Question : " + query.text()
            );

            Collection<ContentRetriever> selection =
                    routeurGemini.route(query);

            if (selection.isEmpty()) {
                System.out.println(
                        "[ROUTAGE] Aucune source sélectionnée"
                );
            }

            if (selection.contains(retrieverRag)) {
                System.out.println(
                        "[ROUTAGE] Source sélectionnée : rag.pdf"
                );
            }

            if (selection.contains(retrieverJava)) {
                System.out.println(
                        "[ROUTAGE] Source sélectionnée : java.txt"
                );
            }

            return selection;
        };

        // 8. Configurer le RAG avancé
        RetrievalAugmentor augmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(routeurAvecLogs)
                        .build();

        // 9. Créer l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(
                        MessageWindowChatMemory.withMaxMessages(10)
                )
                .retrievalAugmentor(augmentor)
                .build();

        System.out.println(
                "\nTest 3 - Routage entre deux sources activé"
        );

        // 10. Conversation interactive
        try (Scanner scanner = new Scanner(System.in)) {

            while (true) {

                System.out.print(
                        "\nVotre question (fin pour quitter) : "
                );

                String question = scanner.nextLine();

                if ("fin".equalsIgnoreCase(question.trim())) {
                    break;
                }

                if (question.isBlank()) {
                    continue;
                }

                String reponse = assistant.chat(question);

                System.out.println(
                        "\nAssistant : " + reponse
                );
            }
        }
    }

    // Créer un retriever pour une source documentaire
    private static ContentRetriever creerRetriever(
            String fichier,
            DocumentParser parser,
            EmbeddingModel embeddingModel) {

        Document document =
                ClassPathDocumentLoader.loadDocument(
                        fichier,
                        parser
                );

        List<TextSegment> segments =
                DocumentSplitters.recursive(300, 30)
                        .split(document);

        EmbeddingStore<TextSegment> store =
                new InMemoryEmbeddingStore<>();

        store.addAll(
                embeddingModel.embedAll(segments).content(),
                segments
        );

        System.out.println(
                "[INDEXATION] " + fichier +
                        " : " + segments.size() + " segments"
        );

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }
}
