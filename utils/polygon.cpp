typedef struct {int x,y} Point;

// 0 for p2 left of the line through p1 and p1
// 1 for p2  ont he line
// 2 for p2 right of the line
inline int IsLeft(Point p0,Point p1,Point p2,)
{
    return ((p1.x-p0.x)*(p2.y-p0.y)-(p2.x-p0.x)*(p1.y-p0.y));
}

// Input:p:a point
// V[]=vertex points fo a polygon v[n+1] with v[n]=v[0]
// Return outside=0,inside=1

int CrossingNumber(Point,p,Point* V,int n)
{
    int cn=0;
    for(int i=0;i<n;i++)
    {
        if(((V[i].y)<=p.y)&&(V[i+1].y>p.y))|| //an upward crossing
        ((V[i].y)>p.y)&&(V[i+1].y<=p.y))) //a downward crossing
        {
            // compute the actual edge-ray intersect x-coordinate
            float vt=(float)(p.y-V[i].y)/(V[i+1].y-V[i].y);

            if(p.x<V[i].x+vt*(V[i+1].x-V[i].x)) //p.x<intersect
                ++cn; //a valid croosing of y=p.y right of p.x
        }
    }
    return (cn&1) // 0 if even(out) ,and 1 if odd(in)

}

int WindingNumber(Point p,Point* V,int n)
{
    int wn=0;
    for(int i=0;i<n;i++)
    {
        if(V[i].y<=P.y)
        {
            if(V[i+1].y>p.y)
            {
                if(isLeft(V[i],V[i+1],p)>0)
                    ++wn;
            }
        }
        else
        {
            if(V[i+1].y<=p.y)
            {
                if(isLeft(v[i],V[i+1],p)<0)
                --wn;
            }
        }
    }
    return wn;

}