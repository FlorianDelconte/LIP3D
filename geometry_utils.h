#include <DGtal/helpers/StdDefs.h>
#include <DGtal/images/ImageContainerBySTLVector.h>
#include <DGtal/shapes/MeshVoxelizer.h>
#include <DGtal/math/linalg/EigenDecomposition.h>
#include <DGtal/io/readers/MeshReader.h>

using namespace DGtal;
using namespace Z3i;
using Image3D = ImageContainerBySTLVector<Z3i::Domain, unsigned int>;

typedef PointVector<9, double> Matrix3x3Point;
typedef SimpleMatrix<double, 3, 3> CoVarianceMat;

CoVarianceMat 
getCoVarianceMatFrom(const Matrix3x3Point &m) {
    CoVarianceMat res;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            res.setComponent(i, j, m[j + i * 3]);
    return res;
}

std::tuple<RealPoint, RealPoint, RealPoint> 
getMainDirsFromVoxels(const Image3D &volImage) {
    Matrix3x3Point cov;
    unsigned int count = 0;
    RealPoint centroid = RealPoint::zero;

    for (const auto &p : volImage.domain()) {
        if (volImage(p) != 0) {
            centroid += RealPoint(p);
            ++count;
        }
    }
    if (count == 0) return {RealPoint::zero, RealPoint::zero, RealPoint::zero};
    centroid /= count;

    for (const auto &p : volImage.domain()) {
        if (volImage(p) != 0) {
            RealPoint d = RealPoint(p) - centroid;
            cov[0] += d[0] * d[0];
            cov[1] += d[0] * d[1];
            cov[2] += d[0] * d[2];
            cov[4] += d[1] * d[1];
            cov[5] += d[1] * d[2];
            cov[8] += d[2] * d[2];
        }
    }
    cov[3] = cov[1]; cov[6] = cov[2]; cov[7] = cov[5];
    cov /= count;

    CoVarianceMat mat = getCoVarianceMatFrom(cov);
    SimpleMatrix<double, 3, 3> eigVecs;
    PointVector<3, double> eigVals;
    DGtal::EigenDecomposition<3, double, CoVarianceMat>::getEigenDecomposition(mat, eigVecs, eigVals);

    std::vector<std::pair<double, int>> v;
    for (int i = 0; i < 3; ++i) v.emplace_back(eigVals[i], i);
    std::sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.first > b.first; });

    return {eigVecs.column(v[0].second), eigVecs.column(v[1].second), eigVecs.column(v[2].second)};
}

Point 
moveCenterAlongDirection(const Point &center, const RealPoint &direction, int offset) {
    RealPoint dir = direction.getNormalized();
    RealPoint offsetVec = dir * offset;
    return center + Point(std::round(offsetVec[0]), std::round(offsetVec[1]), std::round(offsetVec[2]));
}

int 
estimateMaxScanFromDirection(const Image3D &volume, const Z3i::RealPoint &normalDir)
{
    double minProj = std::numeric_limits<double>::max();
    double maxProj = std::numeric_limits<double>::lowest();

    for (auto it = volume.domain().begin(); it != volume.domain().end(); ++it) {
        if (volume(*it) > 0) {
            Z3i::Point p = *it;
            double proj = p[0] * normalDir[0] + p[1] * normalDir[1] + p[2] * normalDir[2];
            minProj = std::min(minProj, proj);
            maxProj = std::max(maxProj, proj);
        }
    }

    return static_cast<int>(std::ceil(maxProj - minProj));
}