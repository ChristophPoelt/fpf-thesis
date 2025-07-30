import lombok.Getter;

@Getter
public class IndividualWithFPF {
    private final double x1;
    private final double x2;
    private final double fpfValue;

    public IndividualWithFPF(Individual individual, double fpfValue) {
        this.x1 = individual.getX1();
        this.x2 = individual.getX2();
        this.fpfValue = fpfValue;
    }
}