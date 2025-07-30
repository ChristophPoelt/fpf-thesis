import lombok.Getter;
import lombok.Setter;
import java.util.ArrayList;
import java.util.Random;
import java.util.BitSet;

@Getter
@Setter
public class Individual {
    private double[] genome;
    private double targetValue;
    private BitSet ancestry;

    public Individual(double[] genome, int initialPopulationSize, int index, Main.ProblemInstance problemInstance) {
        this.genome = genome;
        this.ancestry = new BitSet(initialPopulationSize);
        this.ancestry.set(index);
        calculateTargetValue(problemInstance);
    }

    public Individual(double[] genome, int bitSetSize, Main.ProblemInstance problemInstance) {
        this.genome = genome;
        this.ancestry = new BitSet(bitSetSize);
        calculateTargetValue(problemInstance);
    }

    public void calculateTargetValue(Main.ProblemInstance problemInstance) {
        switch (problemInstance) {
            case SCHWEFEL -> {
                double sum = 0.0;
                for (double xi : genome) {
                    sum += xi * Math.sin(Math.sqrt(Math.abs(xi)));
                }
                this.targetValue = (418.9828872724339 * genome.length - sum) / 10000.0;
            }
            case SCHAFFER -> {
                double sum = 0.0;
                for (int i = 0; i < genome.length - 1; i++) {
                    double xi = genome[i];
                    double xj = genome[i + 1];
                    double term1 = Math.pow(xi * xi + xj * xj, 0.25);
                    double term2 = Math.pow(Math.sin(50 * Math.pow(xi * xi + xj * xj, 0.1)), 2);
                    sum += term1 * (term2 + 1.0);
                }
                this.targetValue = sum / 250;
            }
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Individual [");
        for (int i = 0; i < genome.length; i++) {
            sb.append(String.format("x%d=%.4f", i + 1, genome[i]));
            if (i < genome.length - 1) sb.append(", ");
        }
        sb.append(String.format("] -> Fitness: %.6f", targetValue));
        return sb.toString();
    }


    /**
     * Creates and returns a new individual which is altered only in one dimension by adding
     * a gaussian term with a standard deviation of 1% of the search space.
     *
     * @return New Individual deduced by mutating this Individual
     */
    public Individual mutate(Main.ProblemInstance problemInstance) {
        Random random = new Random();
        double[] mutatedGenome = genome.clone();
        int indexToMutate = random.nextInt(genome.length);

        double mutationStep;
        switch (problemInstance) {
            case SCHWEFEL -> mutationStep = random.nextGaussian() * 10;
            case SCHAFFER -> mutationStep = random.nextGaussian() * 2;
            default -> mutationStep = 0.0;
        }

        mutatedGenome[indexToMutate] += mutationStep;

        // Bounds je nach ProblemInstance
        for (int i = 0; i < mutatedGenome.length; i++) {
            switch (problemInstance) {
                case SCHWEFEL -> mutatedGenome[i] = Math.max(-500, Math.min(500, mutatedGenome[i]));
                case SCHAFFER -> mutatedGenome[i] = Math.max(-100, Math.min(100, mutatedGenome[i]));
            }
        }

        Individual mutated = new Individual(mutatedGenome, ancestry.length(), problemInstance);
        mutated.ancestry.or(this.ancestry);
        return mutated;
    }


    /**
     * Recombines two individuals and returns one of the possible combinations of these parents in an array list
     *
     * @param other The other parent that this parent is recombining with.
     * @return the array list containing one offspring
     */
    public ArrayList<Individual> recombine(Individual other, Main.ProblemInstance problemInstance) {
        Random random = new Random();
        double[] childGenome = new double[genome.length];
        for (int i = 0; i < genome.length; i++) {
            childGenome[i] = (random.nextBoolean() ? genome[i] : other.genome[i]);
        }

        Individual child = new Individual(childGenome, ancestry.length(), problemInstance);
        child.ancestry.or(this.ancestry);
        child.ancestry.or(other.ancestry);

        ArrayList<Individual> offspring = new ArrayList<>();
        offspring.add(child);
        return offspring;
    }
}