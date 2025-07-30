import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class EvolutionaryAlgorithm {
    private final int POPULATION_SIZE;
    private final int GENERATIONS;
    private final double MUTATION_RATE;
    private final double RECOMBINATION_RATE;
    private final double MIGRATION_RATE;
    private final Main.ProblemInstance problemInstance;

    private Population population;
    private Population initialPopulation;

    private ArrayList<Double> bestFitnessHistory = new ArrayList<>(); // only used for improvement-based stopping criterion
    private List<Individual> bestIndividualsPerGeneration = new ArrayList<>();
    private final int SLIDING_WINDOW_SIZE = 10; // Number of generations in the sliding window
    private final double EPSILON = 0.00001; // Threshold for average improvement

    private final double DELTA = 0.01; // Threshold for convergence

    private final double CD = 0.01; // Threshold for diversity

    public enum StoppingCriterion {FIXED_NUMBER_OF_GENERATIONS, IMPROVEMENT_BASED_STOPPING_CRITERION,
        FITNESS_BASED_STOPPING_CRITERION, DIVERSITY_BASED_STOPPING_CRITERION, FIXED_TARGET_VALUE}

    private static StoppingCriterion stoppingCriterion = StoppingCriterion.FIXED_NUMBER_OF_GENERATIONS;
    //private static StoppingCriterion stoppingCriterion = StoppingCriterion.IMPROVEMENT_BASED_STOPPING_CRITERION;
    //private static StoppingCriterion stoppingCriterion = StoppingCriterion.FITNESS_BASED_STOPPING_CRITERION;
    //private static StoppingCriterion stoppingCriterion = StoppingCriterion.DIVERSITY_BASED_STOPPING_CRITERION;
    //private static StoppingCriterion stoppingCriterion = StoppingCriterion.FIXED_TARGET_VALUE;

    EvolutionaryAlgorithm (int populationSize, int generations, double mutationRate,
                           double recombinationRate, double migrationRate, Main.ProblemInstance problemInstance) {
        POPULATION_SIZE = populationSize;
        GENERATIONS = generations;
        MUTATION_RATE = mutationRate;
        RECOMBINATION_RATE = recombinationRate;
        MIGRATION_RATE = migrationRate;
        this.problemInstance = problemInstance;
    }

    public RunResult runEvolutionaryAlgorithmLoop() {
        Population firstGeneration = new Population(POPULATION_SIZE, problemInstance);
        firstGeneration.getIndividuals().sort((i1, i2) -> Double.compare(i1.getTargetValue(), i2.getTargetValue()));
        this.initialPopulation = new Population(firstGeneration, problemInstance);
        this.population = firstGeneration;
        int generation = 0;
        logGeneration(generation);
        while (generation < GENERATIONS) {
            population.getIndividuals().addAll(applyEvolutionaryOperators());
            cutOffSelection();

            if (shouldStop(generation)) {
                System.out.println(stoppingCriterion + " met at generation " + generation);
                break;
            }

            logGeneration(generation);
            trackBestIndividual();
            generation++;
        }

        List<IndividualWithFPF> individualsWithFPF = calculateFPF();
        printBestIndividualsHistory();
        double finalBestFitness = population.getIndividuals().stream()
                .mapToDouble(Individual::getTargetValue)
                .min()
                .orElse(Double.MAX_VALUE);
        return new RunResult(individualsWithFPF, generation, finalBestFitness, bestIndividualsPerGeneration);
    }

    private boolean shouldStop(int generation) {
        switch (stoppingCriterion) {
            case IMPROVEMENT_BASED_STOPPING_CRITERION:
                return checkImprovementStopping(generation);
            case FITNESS_BASED_STOPPING_CRITERION:
                return checkFitnessBasedStoppingCriterion();
            case DIVERSITY_BASED_STOPPING_CRITERION:
                return checkDiversityBasedStoppingCriterion();
            case FIXED_TARGET_VALUE:
                return checkFixedTargetValue();
            default:
                return false; // FIXED_NUMBER_OF_GENERATIONS case, which runs until max generations
        }
    }

    private void logGeneration(int generation) {
        System.out.printf("\n=== Generation %d ===\n", generation);
        for (Individual ind : population.getIndividuals()) {
            System.out.printf("Individual: [x1=%8.2f, x2=%8.2f] -> Fitness: %.6f, Ancestry: %s\n",
                    ind.getX1(), ind.getX2(), ind.getTargetValue(), ind.getAncestry());
        }
        double bestFitness = population.getIndividuals().stream()
                .mapToDouble(Individual::getTargetValue)
                .min()
                .orElse(Double.MAX_VALUE);
        System.out.printf("Generation %d: Best Fitness = %.6f%n", generation, bestFitness);
    }

    private boolean checkImprovementStopping(int generation) {
        double bestFitness = population.getIndividuals().stream()
                .mapToDouble(Individual::getTargetValue)
                .min()
                .orElse(Double.MAX_VALUE);
        bestFitnessHistory.add(bestFitness);

        if (bestFitnessHistory.size() >= SLIDING_WINDOW_SIZE) {
            boolean shouldStop = checkImprovementBasedStoppingCriterion();
            bestFitnessHistory.remove(0);
            return shouldStop;
        }
        return false;
    }

    public ArrayList<Individual> applyEvolutionaryOperators() {
        int mutationCount = (int) Math.ceil(MUTATION_RATE*POPULATION_SIZE);
        int recombinationCount = (int) Math.ceil(RECOMBINATION_RATE*POPULATION_SIZE);
        int migrationCount = (int) Math.ceil(MIGRATION_RATE*POPULATION_SIZE);

        //Mutation
        Collections.shuffle(population.getIndividuals());
        ArrayList<Individual> mutantOffspring = new ArrayList<Individual>();
        for(int i=0; i<mutationCount; i++) {
            mutantOffspring.add(population.getIndividuals().get(i).mutate(problemInstance));
        }

        //Recombination
        Collections.shuffle(population.getIndividuals());
        ArrayList<Individual> recombinationOffspring = new ArrayList<Individual>();
        for(int i=0; i<recombinationCount; i++) {
            recombinationOffspring.addAll(population.getIndividuals().get(i).recombine(
                    population.getIndividuals().get(POPULATION_SIZE-i-1), problemInstance));
        }

        //Migration
        ArrayList<Individual> migrationOffspring = new ArrayList<Individual>();
        for(int i=0; i<migrationCount; i++) {
            migrationOffspring.add(migrate());
        }

        mutantOffspring.addAll(recombinationOffspring);
        mutantOffspring.addAll(migrationOffspring);
        return mutantOffspring;
    }

    public Individual migrate() {
        switch (problemInstance) {
            case SCHWEFEL:
                double randomX1Schwefel = (Math.random() * 1000) - 500;
                double randomX2Schwefel = (Math.random() * 1000) - 500;
                return new Individual(randomX1Schwefel, randomX2Schwefel, 50, problemInstance);
            case HIMMELBLAU:
                double randomX1Himmelblau = (Math.random() * 12) - 10;
                double randomX2Himmelblau = (Math.random() * 12) - 10;
                return new Individual(randomX1Himmelblau, randomX2Himmelblau, 50, problemInstance);
            case H1, SCHAFFER:
                double randomX1H1 = (Math.random() * 200) - 100;
                double randomX2H1 = (Math.random() * 200) - 100;
                return new Individual(randomX1H1, randomX2H1, 50, problemInstance);
            default:
                return null;
        }
    }

    public List<IndividualWithFPF> calculateFPF() {
        int initialPopulationSize = initialPopulation.getIndividuals().size();
        List<IndividualWithFPF> individualsWithFPF = new ArrayList<>();

        System.out.println("\n=== Final Productive Fitness (FPF) for Initial Generation ===");

        for (int i = 0; i < initialPopulationSize; i++) {
            Individual initialInd = initialPopulation.getIndividuals().get(i);
            int count = 0;
            double totalFitness = 0.0;

            // Check descendants in final population
            for (Individual finalIndividual : population.getIndividuals()) {
                if (finalIndividual.getAncestry().get(i)) {
                    totalFitness += finalIndividual.getTargetValue();
                    count++;
                    // Debugging output
                    System.out.printf("Descendant found: Initial Individual %d contributes to %s\n", i, finalIndividual);
                }
            }

            // Compute FPF value
            double fpfValue = (count > 0) ? totalFitness / count : 1.0;
            individualsWithFPF.add(new IndividualWithFPF(initialInd, fpfValue));

            System.out.printf("Initial Individual %2d: [x1=%8.2f, x2=%8.2f] -> FPF: %.6f (Descendants: %d)%n",
                    i, initialInd.getX1(), initialInd.getX2(), fpfValue, count);
        }

        return individualsWithFPF;
    }

    public void cutOffSelection() {
        population.getIndividuals().sort((i1, i2) -> Double.compare(i1.getTargetValue(), i2.getTargetValue()));

        while (population.getIndividuals().size() > POPULATION_SIZE) {
            population.getIndividuals().remove(population.getIndividuals().size() - 1);
        }
    }

    public boolean checkImprovementBasedStoppingCriterion() {
        int size = bestFitnessHistory.size();
        if (size < SLIDING_WINDOW_SIZE) {
            return false; // Not enough data to evaluate
        }

        // Calculate the average improvement over the sliding window
        double totalImprovement = 0.0;
        for (int i = 1; i < size; i++) {
            double improvement = Math.abs(bestFitnessHistory.get(i) - bestFitnessHistory.get(i - 1));
            totalImprovement += improvement;
        }
        double averageImprovement = totalImprovement / (size - 1);

        // Check if the average improvement is below the threshold
        return averageImprovement < EPSILON;
    }

    public boolean checkFitnessBasedStoppingCriterion() {
        // Get the best and worst fitness values in the population
        double fBest = population.getIndividuals().stream()
                .mapToDouble(Individual::getTargetValue)
                .min()
                .orElse(Double.MAX_VALUE); // Best fitness value (smallest)
        double fWorst = population.getIndividuals().stream()
                .mapToDouble(Individual::getTargetValue)
                .max()
                .orElse(Double.MIN_VALUE); // Worst fitness value (largest)

        // Check if the difference is below the threshold
        return (fWorst - fBest) < DELTA;
    }

    public boolean checkDiversityBasedStoppingCriterion() {
        double diversity = calculateDiversity();
        System.out.println("Current Diversity: " + diversity);
        return diversity < CD;
    }

    public double calculateDiversity() {
        List<Individual> individuals = population.getIndividuals();
        int n = individuals.size();
        if (n <= 1) return 0.0; // Keine Diversität bei weniger als 2 Individuen

        double totalDistance = 0.0;
        int pairCount = 0;

        // Berechnung der paarweisen Distanzen
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                Individual ind1 = individuals.get(i);
                Individual ind2 = individuals.get(j);

                // Paarweise euklidische Distanz
                double distance = Math.sqrt(
                        Math.pow(ind1.getX1() - ind2.getX1(), 2) +
                                Math.pow(ind1.getX2() - ind2.getX2(), 2)
                );

                totalDistance += distance;
                pairCount++;
            }
        }

        // Durchschnittliche paarweise Distanz
        return totalDistance / pairCount;
    }

    public boolean checkFixedTargetValue() {
        switch (problemInstance) {
            case SCHWEFEL:
                return population.getIndividuals().stream()
                        .anyMatch(ind -> ind.getTargetValue() < 0.0001);
            case HIMMELBLAU:
                return population.getIndividuals().stream()
                        .anyMatch(ind -> ind.getTargetValue() < 0.00005);
            case H1:
                return population.getIndividuals().stream()
                        .anyMatch(ind -> ind.getTargetValue() < 0.5);
            // adapt this value after normalizing the Schaffer Function
            case SCHAFFER:
                return population.getIndividuals().stream()
                        .anyMatch(ind -> ind.getTargetValue() < 0.1);
            default:
                return false;
        }
    }

    private void trackBestIndividual() {
        Individual bestIndividual = population.getIndividuals().stream()
                .min((i1, i2) -> Double.compare(i1.getTargetValue(), i2.getTargetValue()))
                .orElse(null); // If empty, return null (shouldn't happen)

        if (bestIndividual != null) {
            bestIndividualsPerGeneration.add(bestIndividual);
        }
    }

    public void printBestIndividualsHistory() {
        System.out.println("\n=== Best Individuals of Each Generation ===");
        for (int i = 0; i < bestIndividualsPerGeneration.size(); i++) {
            Individual best = bestIndividualsPerGeneration.get(i);
            System.out.printf("Generation %3d: [x1=%8.2f, x2=%8.2f] -> Fitness: %.6f%n",
                    i, best.getX1(), best.getX2(), best.getTargetValue());
        }
    }
}