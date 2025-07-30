import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

    private static final int POPULATION_SIZE = 50;
    private static final int GENERATIONS = 200;
    private static final double MUTATION_RATE = 0.1;
    private static final double RECOMBINATION_RATE = 0.15;
    private static final double MIGRATION_RATE = 0.1;
    private static final int RUNS = 500;
    private static final int GENOME_LENGTH = 10;

    public enum ProblemInstance {
        SCHWEFEL, SCHAFFER
    }

    public static void main(String[] args) {

        //ProblemInstance selectedProblem = ProblemInstance.SCHWEFEL;
        ProblemInstance selectedProblem = ProblemInstance.SCHAFFER;

        runExperiment(selectedProblem);
    }

    private static void runExperiment(ProblemInstance problem) {
        List<RunResult> results = new ArrayList<>();

        for (int i = 0; i < RUNS; i++) {
            EvolutionaryAlgorithm ea = new EvolutionaryAlgorithm(
                    POPULATION_SIZE,
                    GENERATIONS,
                    MUTATION_RATE,
                    RECOMBINATION_RATE,
                    MIGRATION_RATE,
                    GENOME_LENGTH,
                    problem
            );
            RunResult result = ea.runEvolutionaryAlgorithmLoop();
            results.add(result);
            System.out.println("Completed run " + (i + 1) + ", generations: " + result.getTotalGenerations());
        }

        saveResultsToJson(results, problem.name().toLowerCase() + "_10D_ImprovementBased.json");

        double avgGenerations = results.stream()
                .mapToInt(RunResult::getTotalGenerations)
                .average()
                .orElse(0);

        double avgFinalFitness = results.stream()
                .mapToDouble(RunResult::getFinalBestFitness)
                .average()
                .orElse(0);

        System.out.printf("\nDurchschnittliche Generationen: %.2f%n", avgGenerations);
        System.out.printf("Durchschnittliche finale Fitness: %.6f%n", avgFinalFitness);
    }

    private static void saveResultsToJson(List<RunResult> results, String fileName) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(fileName), results);
            System.out.println("Saved results to " + fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}