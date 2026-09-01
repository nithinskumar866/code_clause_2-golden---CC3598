using System.Buffers.Binary;

namespace JobPortal.Api.Services.Vectors;

public static class VectorMath
{
    /// <summary>
    /// Scales a vector to unit length in place and returns it. Every vector is
    /// normalised before it is stored, which turns cosine similarity into a plain
    /// dot product at search time — the same answer, without renormalising the
    /// whole corpus on every query.
    /// </summary>
    public static float[] Normalize(float[] vector)
    {
        double sumOfSquares = 0;
        for (var i = 0; i < vector.Length; i++) sumOfSquares += (double)vector[i] * vector[i];

        var norm = Math.Sqrt(sumOfSquares);
        // An all-zero vector has no direction; leaving it alone yields a similarity
        // of 0 against everything, which is the correct reading of "no signal".
        if (norm <= 1e-12) return vector;

        for (var i = 0; i < vector.Length; i++) vector[i] = (float)(vector[i] / norm);
        return vector;
    }

    /// <summary>Cosine similarity of two already-normalised vectors.</summary>
    public static double Dot(float[] a, float[] b)
    {
        if (a.Length != b.Length) return 0;
        double sum = 0;
        for (var i = 0; i < a.Length; i++) sum += (double)a[i] * b[i];
        return sum;
    }

    /// <summary>Cosine similarity that does not assume normalisation.</summary>
    public static double Cosine(float[] a, float[] b)
    {
        if (a.Length != b.Length || a.Length == 0) return 0;
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        if (na <= 1e-12 || nb <= 1e-12) return 0;
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    // Explicit little-endian rather than BitConverter's platform-dependent order:
    // these bytes are persisted, and a database written on one machine must read
    // back identically on another.

    public static byte[] ToBytes(float[] vector)
    {
        var bytes = new byte[vector.Length * sizeof(float)];
        for (var i = 0; i < vector.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * sizeof(float)), vector[i]);
        }
        return bytes;
    }

    public static float[] FromBytes(byte[] bytes)
    {
        var count = bytes.Length / sizeof(float);
        var vector = new float[count];
        for (var i = 0; i < count; i++)
        {
            vector[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * sizeof(float)));
        }
        return vector;
    }
}
