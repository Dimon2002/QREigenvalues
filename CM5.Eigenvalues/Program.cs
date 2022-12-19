namespace CM5.Eigenvalues;

public static class VectorProjection
{
    public static List<double> CalcProjection(List<double> Vector1, List<double> Vector2) =>
            ScalarToVector(ScalarVector(Vector1, Vector2) / ScalarVector(Vector2, Vector2), Vector2);
    public static double ScalarVector(List<double> Vector1, List<double> Vector2) =>
            Vector1.Count == Vector2.Count
            ? Vector1.Zip(Vector2, (x, y) => x * y).Sum(item => item + item)
            : 0;
    public static List<double> ScalarToVector(double value, List<double> vector) =>
        new(vector.Select(item => item * value));
}

public static class DecompositionToColumn
{
    public static List<double[]> MatrixToColumnVectors(double[][] a)
    {
        List<double[]> VectorColumns = new();

        var aSize = a.GetLength(0);

        for (int i = 0; i < aSize; ++i)
        {
            List<double> Column = new();
            for (int j = 0; j < aSize; ++j)
                Column.Add(a[j][i]);
            VectorColumns.Add(Column.ToArray());
            Column.Clear();
        }
        return VectorColumns;
    }
}

public static class NormOfVector
{
    public static double CalcNorm(List<double> v) =>
        Math.Sqrt(v.Sum(item => item * item));
}

public static class QRDecomposition
{
    public static List<double[][]> Decomposition(double[][] a)
    {
        List<double[]> listColumns = new();

        foreach (var item in DecompositionToColumn.MatrixToColumnVectors(a))
            listColumns.Add(item);

        List<double[]> u = new();
        List<double[]> e = new();

        u.Add(listColumns[0]);
        e.Add(VectorProjection.ScalarToVector(1 / NormOfVector.CalcNorm(u[0].ToList()), u[0].ToList()).ToArray());

        for (int i = 1; i < a.GetLength(0); i++)
        {
            List<double> projAcc = new(new double[a.GetLength(0)]);

            for (int j = 0; j < i; ++j)
            {
                List<double> proji = VectorProjection.CalcProjection(listColumns[i].ToList(), e[j].ToList());

                for (int k = 0; k < projAcc.Count; k++)
                    projAcc[k] += proji[k];
                // projAcc = (List<double>)projAcc.Zip(proji, (x, y) => x + y);
            }

            List<double> ui = new(new double[a.GetLength(0)]);

            for (int j = 0; j < ui.Count; ++j)
                ui[j] = a[j][i] - projAcc[j];
            u.Add(ui.ToArray());
            e.Add(VectorProjection.ScalarToVector(1 / NormOfVector.CalcNorm(ui), ui).ToArray());
        }


        List<double[]> Q = new(a.GetLength(0));

        for (int i = 0; i < Q.Capacity; ++i)
        {
            Q.Add(new double[a.GetLength(0)]);
            for (int j = 0; j < Q[i].Length; ++j)
                Q[i][j] = e[j][i];
        }

        List<double[]> R = new(a.GetLength(0));
        for (int i = 0; i < R.Capacity; ++i)
        {
            R.Add(new double[a.GetLength(0)]);
            for (int j = 0; j < R[i].Length; ++j)
            {
                if (i >= j)
                {
                    R[i][j] = VectorProjection.ScalarVector(e[j].ToList(), listColumns[i].ToList());
                    continue;
                }
                R[i][j] = 0;
            }
        }

        List<double[][]> res = new();

        Q = Transpose(Q);
        res.Add(Multiplication(Q, a).ToArray());
        res.Add(Transpose(Q).ToArray());
        return res;
    }
    private static List<double[]> Transpose(List<double[]> matrix)
    {
        for (int i = 0; i < matrix.Count; ++i)
            for (int j = 0; j < i; ++j)
                (matrix[i][j], matrix[j][i]) = (matrix[j][i], matrix[i][j]);
        return matrix;
    }
    private static List<double[]> Multiplication(List<double[]> Q, double[][] a)
    {
        List<double[]> res = new(a.GetLength(0));

        for (int i = 0; i < res.Capacity; i++)
            res.Add(new double[a.GetLength(0)]);

        for (int i = 0; i < Q.Count; i++)
            for (int j = 0; j < a.GetLength(0); j++)
                for (int k = 0; k < a.GetLength(0); k++)
                    res[i][j] += Q[i][k] * a[k][j];

        for (int i = 0; i < res.Count; i++)
            for (int j = 0; j < res[i].Length; j++)
                if (res[i][j] < 1e-14) res[i][j] = 0;
        return res;
    }
}

public static class EigenValuesExtractionQR
{
    public static List<double[][]> EigenValues(in double[][] a, double accuracy, int maxIterations)
    {
        var aOnIteration = a;
        double[][] q = null;

        for (int i = 0; i < maxIterations; i++)
        {
            IList<double[][]> qr = QRDecomposition.Decomposition(aOnIteration);
            aOnIteration = MatrixMultiplicationProduct(qr[0], qr[1]);

            if (q == null)
            {
                q = qr[0];
                continue;
            }

            double[][] qNew = MatrixMultiplicationProduct(qr[0], q);
            bool accuracyAcheived = true;
            for (int n = 0; n < q.Length; n++)
            {
                for (int m = 0; m < q[n].Length; m++)
                {
                    if (Math.Abs(Math.Abs(qNew[n][m]) - Math.Abs(q[n][m])) > accuracy)
                    {
                        accuracyAcheived = false;
                        break;
                    }
                }
                if (!accuracyAcheived)
                    break;
            }
            q = qNew;
            if (accuracyAcheived)
                break;
        }

        if (q == null)
            return new List<double[][]>() { };

        List<double[][]> res = new()
        {
            q,
            aOnIteration
        };
        return res;
    }

    private static double[][] MatrixMultiplicationProduct(double[][] a, double[][] b)
    {
        double[][] res = new double[a.Length][];
        for (int i = 0; i < res.GetLength(0); i++)
            res[i] = new double[a.Length];

        for (int i = 0; i < a.GetLength(0); i++)
            for (int j = 0; j < b.GetLength(0); j++)
                for (int k = 0; k < b.GetLength(0); k++)
                    res[i][j] += a[i][k] * b[k][j];
        return res;
    }
}

public class Program
{
    static void Main()
    {
        double[][] a = new double[][]
        {
            new double[] {1, 2, 2},
            new double[] {2, 2, 2},
            new double[] {2, 2, 3}
        };

        var x = EigenValuesExtractionQR.EigenValues(a, 0.1, 1000);

        List<double> EigenValues = new();

        for (int i = 0; i < x[1].GetLength(0); i++)
            EigenValues.Add(x[1][i][i]);

        EigenValues.Sort();

        double maxEigenValue = EigenValues.Max();
        double minEigenValue = EigenValues.Min();

        Console.WriteLine($"Max = {Math.Round(maxEigenValue,4)}\n" +
                          $"Min = {Math.Round(minEigenValue,4)}");
    }
}