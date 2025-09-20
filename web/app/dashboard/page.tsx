import LiveGlobe from "@/components/LiveGlobe";


export default function Home() {

    return (
      <div>
        <LiveGlobe simHoursPerSec={24} />
      </div>
    );
  }