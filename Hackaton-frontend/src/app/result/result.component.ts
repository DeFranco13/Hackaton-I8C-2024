import { Component, OnInit, OnDestroy } from '@angular/core';
import { Subscription } from 'rxjs';
import { ApiService } from '../service/api.service';

@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.css']
})
export class ResultComponent implements OnInit, OnDestroy {
    loading = true;
    anomalies: any = [];
    displayData: any = [];
    private logInterval?: any;
    private dataSubscription?: Subscription;

    constructor(private apiService: ApiService) {}

    ngOnInit() {
        this.dataSubscription = this.apiService.getData().subscribe({
            next: (data) => {
                this.anomalies = data;
                this.displayData = Object.entries(data).map(([key, value]) => ({ key, value }));
                this.loading = false;
            },
            error: (error) => {
                console.error('Error fetching data', error);
                this.loading = false;
            }
        });

        this.logInterval = setInterval(() => {
            console.log(this.displayData);
        }, 10000);  // Logs the displayData every 10 seconds
    }

    ngOnDestroy() {
        // Clean up the interval and subscription
        if (this.logInterval) {
            clearInterval(this.logInterval);
        }
        if (this.dataSubscription) {
            this.dataSubscription.unsubscribe();
        }
    }
}
