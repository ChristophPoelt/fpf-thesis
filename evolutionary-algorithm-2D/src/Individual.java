import lombok.Getter;
import lombok.Setter;
import java.util.ArrayList;
import java.util.Random;
import java.util.BitSet;

@Getter
@Setter
public class Individual {
    private double x1;
    private double x2;
    private double targetValue;
    private BitSet ancestry;

    public Individual(double x, double x2, int initialPopulationSize, int index, Main.ProblemInstance problemInstance) {
        this.x1 = x;
        this.x2 = x2;
        this.ancestry = new BitSet(initialPopulationSize);
        this.ancestry.set(index);
        calculateTargetValue(problemInstance);
    }

    public Individual(double x, double x2, int bitSetSize, Main.ProblemInstance problemInstance) {
        this.x1 = x;
        this.x2 = x2;
        this.ancestry = new BitSet(bitSetSize);
        calculateTargetValue(problemInstance);
    }

    public void calculateTargetValue(Main.ProblemInstance problemInstance) {
        switch (problemInstance) {
            //max value at about 13000. normalization is done by dividing with 13000
            case HIMMELBLAU:
                this.targetValue = (Math.pow(Math.pow(x1, 2) + x2 - 11, 2) + Math.pow(x1 + Math.pow(x2, 2) - 7, 2))/13000;
                break;
            //max value at about 2. normalization is done by dividing with 2
            case H1:
                double num = Math.pow(Math.sin(x1 - x2 / 8.0), 2) + Math.pow(Math.sin(x2 + x1 / 8.0), 2);
                double denum = Math.sqrt(Math.pow(x1 - 8.6998, 2) + Math.pow(x2 - 6.7665, 2)) + 1;
                double h1Value = num / denum;

                this.targetValue = (- h1Value + 2)/2; // Convert to minimization problem and normalize
                break;
            case SCHWEFEL:
                double rawTargetValue = 2 * 418.982887272433 -
                        (x1 * Math.sin(Math.sqrt(Math.abs(x1))) + x2 * Math.sin(Math.sqrt(Math.abs(x2))));
                this.targetValue = (rawTargetValue / 4000.0);
                break;
            case SCHAFFER:
                double term1 = Math.pow(x1 * x1 + x2 * x2, 0.25);
                double term2 = Math.sin(50 * Math.pow(x1 * x1 + x2 * x2, 0.10));
                this.targetValue = (term1 * (term2 * term2 + 1.0))/25;
                break;
        }
    }

    @Override
    public String toString() {
        return "Individual [" + x1 + "; " + x2 + "] with Target Value: " ;
    }


    /**
     * Creates and returns a new individual which is altered only in one dimension by adding
     * a gaussian term with a standard deviation of 1% of the search space.
     *
     * @return New Individual deduced by mutating this Individual
     */
    public Individual mutate(Main.ProblemInstance problemInstance) {
        Random random = new Random();
        double mutateTerm = 1;
        //Standard deviation of should be about 1% of the search space
        switch (problemInstance) {
            case SCHWEFEL:
                mutateTerm = random.nextGaussian() * 10;
                break;
            case HIMMELBLAU:
                mutateTerm = random.nextGaussian() * 0.12;
                break;
            case H1, SCHAFFER:
                mutateTerm = random.nextGaussian() * 2;
                break;
            default:
                break;
        }

        Individual mutated;

        double newX1 = x1;
        double newX2 = x2;

        if (random.nextBoolean()) {
            newX1 = x1 + mutateTerm;
        } else {
            newX2 = x2 + mutateTerm;
        }

        // Ensure boundaries are respected
        switch (problemInstance) {
            case SCHWEFEL:
                newX1 = Math.max(-500, Math.min(500, newX1));
                newX2 = Math.max(-500, Math.min(500, newX2));
                break;
            case HIMMELBLAU:
                newX1 = Math.max(-6, Math.min(6, newX1));
                newX2 = Math.max(-6, Math.min(6, newX2));
                break;
            case H1, SCHAFFER:
                newX1 = Math.max(-100, Math.min(100, newX1));
                newX2 = Math.max(-100, Math.min(100, newX2));
                break;
            default:
                break;
        }

        mutated = new Individual(newX1, newX2, ancestry.length(), problemInstance);
        mutated.ancestry.or(this.ancestry);

        System.out.println("Mutation:");
        System.out.println("Parent Ancestry: " + this.ancestry);
        System.out.println("Mutated Ancestry: " + mutated.ancestry);

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
        ArrayList<Individual> offspring = new ArrayList<>();
        Individual child1 = new Individual(x1, other.x2, ancestry.length(), problemInstance);
        Individual child2 = new Individual(other.x1, x2, ancestry.length(), problemInstance);

        child1.ancestry.or(this.ancestry);
        child1.ancestry.or(other.ancestry);
        child2.ancestry.or(this.ancestry);
        child2.ancestry.or(other.ancestry);

        System.out.println("Recombination:");

        if (random.nextBoolean()) {
            offspring.add(child1);
            System.out.println("Parent 1 Ancestry: " + this.ancestry);
            System.out.println("Child 1 Ancestry: " + child1.ancestry);
        } else {
            offspring.add(child2);
            System.out.println("Parent 2 Ancestry: " + other.ancestry);
            System.out.println("Child 2 Ancestry: " + child2.ancestry);
        }
        //only one child is returned
        //offspring.add(child1);
        //offspring.add(child2);
        return offspring;
    }
}