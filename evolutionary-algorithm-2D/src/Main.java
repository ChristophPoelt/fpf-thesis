import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

    private static final int POPULATION_SIZE = 50;
    private static final int GENERATIONS = 50; // upper limit for the number of generations
    private static final double MUTATION_RATE = 0.1;
    private static final double RECOMBINATION_RATE = 0.15;
    private static final double MIGRATION_RATE = 0.1;
    private static final int RUNS = 500;
    private static final String JSON_FILE = "_schwefelFixedGenerations.json";

    public enum ProblemInstance {
        SCHWEFEL, HIMMELBLAU, H1, SCHAFFER
    }

    public static void main(String[] args) {
        ProblemInstance problem = ProblemInstance.SCHWEFEL;
        //ProblemInstance problem = ProblemInstance.HIMMELBLAU;
        //ProblemInstance problem = ProblemInstance.H1;
        //ProblemInstance problem = ProblemInstance.SCHAFFER;


        List<RunResult> results = new ArrayList<>();

        for (int i = 0; i < RUNS; i++) {
            EvolutionaryAlgorithm ea = new EvolutionaryAlgorithm(POPULATION_SIZE, GENERATIONS, MUTATION_RATE, RECOMBINATION_RATE, MIGRATION_RATE, problem);
            RunResult result = ea.runEvolutionaryAlgorithmLoop();
            results.add(result);
            System.out.println("Completed run " + (i + 1) + " with " + result.getTotalGenerations() + " generations.");
        }

        saveResultsToJson(results, JSON_FILE);

        double avgGenerations = results.stream().mapToInt(RunResult::getTotalGenerations).average().orElse(0);
        double avgFinalFitness = results.stream().mapToDouble(RunResult::getFinalBestFitness).average().orElse(0);
        System.out.printf("\nAverage Generations: %.2f\n", avgGenerations);
        System.out.printf("Average Final Best Fitness: %.6f\n", avgFinalFitness);
    }

    private static void saveResultsToJson(List<RunResult> results, String fileName) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(fileName), results);
            System.out.println("Run results saved to " + fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}